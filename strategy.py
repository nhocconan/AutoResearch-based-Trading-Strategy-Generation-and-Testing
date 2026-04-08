#!/usr/bin/env python3
# 6h_keltner_breakout_volume_v1
# Hypothesis: Combines 6h Keltner Channel breakouts with 12h trend direction (SMA50 slope) and volume confirmation.
# Long when price breaks above upper Keltner Channel, 12h SMA50 slope > 0, and volume > 1.5x average.
# Short when price breaks below lower Keltner Channel, 12h SMA50 slope < 0, and volume > 1.5x average.
# Exit when price re-enters the Keltner Channel or volume drops below average.
# Uses tight entry conditions to limit trades and reduce fee drag. Target: 20-40 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_keltner_breakout_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h Keltner Channel (20, 1.5)
    keltner_period = 20
    atr_period = 20
    keltner_mult = 1.5
    
    # Calculate ATR
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(alpha=1/atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Calculate EMA of typical price for Keltner middle line
    typical_price = (high + low + close) / 3
    ema_typical = pd.Series(typical_price).ewm(span=keltner_period, adjust=False, min_periods=keltner_period).mean().values
    
    # Upper and lower Keltner bands
    keltner_upper = ema_typical + keltner_mult * atr
    keltner_lower = ema_typical - keltner_mult * atr
    
    # Volume filter: 1.5x 20-period average
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period-1, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period+1:i+1])
    
    vol_surge = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 1.5 * vol_ma[i]
    
    # Get 12h data for trend direction (SMA50 slope)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    sma_period = 50
    sma50_12h = pd.Series(close_12h).rolling(window=sma_period, min_periods=sma_period).mean().values
    # Calculate slope: positive if current SMA > SMA 3 periods ago
    sma50_slope_12h = np.full(len(close_12h), np.nan)
    for i in range(3, len(close_12h)):
        if not np.isnan(sma50_12h[i]) and not np.isnan(sma50_12h[i-3]):
            sma50_slope_12h[i] = sma50_12h[i] - sma50_12h[i-3]
    sma50_slope_12h_aligned = align_htf_to_ltf(prices, df_12h, sma50_slope_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(keltner_period, atr_period, vol_ma_period, 3) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(keltner_upper[i]) or np.isnan(keltner_lower[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(sma50_slope_12h_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price below upper Keltner band or volume drops below average
            if close[i] < keltner_upper[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price above lower Keltner band or volume drops below average
            if close[i] > keltner_lower[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price above upper Keltner band, 12h SMA50 slope positive, volume surge
            if (close[i] > keltner_upper[i] and 
                sma50_slope_12h_aligned[i] > 0 and 
                vol_surge[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Price below lower Keltner band, 12h SMA50 slope negative, volume surge
            elif (close[i] < keltner_lower[i] and 
                  sma50_slope_12h_aligned[i] < 0 and 
                  vol_surge[i]):
                position = -1
                signals[i] = -0.25
    
    return signals