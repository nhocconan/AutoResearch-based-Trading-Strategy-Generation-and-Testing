#!/usr/bin/env python3
# 4h_volatility_breakout_v1
# Hypothesis: Combines 4h ATR-based volatility breakout with 1d trend direction (SMA50 slope) and volume confirmation.
# Long when: price > upper ATR band, 1d SMA50 slope > 0, volume > 1.5x average.
# Short when: price < lower ATR band, 1d SMA50 slope < 0, volume > 1.5x average.
# Exit when price returns to SMA50 or volume drops below average.
# Uses volatility breakout to capture momentum with volatility filter to avoid false signals.
# Target: 20-40 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_volatility_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h ATR for volatility bands
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr = np.zeros(n)
    atr[atr_period-1] = np.mean(tr[:atr_period])
    for i in range(atr_period, n):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # 4h SMA50 for mean reversion exit
    sma_period = 50
    close_series = pd.Series(close)
    sma50 = close_series.rolling(window=sma_period, min_periods=sma_period).mean().values
    
    # ATR multiplier for bands
    atr_mult = 2.0
    upper_band = sma50 + atr_mult * atr
    lower_band = sma50 - atr_mult * atr
    
    # Volume filter: 1.5x 20-period average
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period-1, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period+1:i+1])
    
    vol_surge = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 1.5 * vol_ma[i]
    
    # Get 1d data for trend direction (SMA50 slope)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    sma50_1d = pd.Series(close_1d).rolling(window=sma_period, min_periods=sma_period).mean().values
    # Calculate slope: positive if current SMA > SMA 3 periods ago
    sma50_slope_1d = np.full(len(close_1d), np.nan)
    for i in range(3, len(close_1d)):
        if not np.isnan(sma50_1d[i]) and not np.isnan(sma50_1d[i-3]):
            sma50_slope_1d[i] = sma50_1d[i] - sma50_1d[i-3]
    sma50_slope_1d_aligned = align_htf_to_ltf(prices, df_1d, sma50_slope_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(atr_period, sma_period, vol_ma_period, 3) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(sma50[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(sma50_slope_1d_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price below SMA50 or volume drops below average
            if close[i] < sma50[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price above SMA50 or volume drops below average
            if close[i] > sma50[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price above upper ATR band, 1d SMA50 slope positive, volume surge
            if (close[i] > upper_band[i] and 
                sma50_slope_1d_aligned[i] > 0 and 
                vol_surge[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Price below lower ATR band, 1d SMA50 slope negative, volume surge
            elif (close[i] < lower_band[i] and 
                  sma50_slope_1d_aligned[i] < 0 and 
                  vol_surge[i]):
                position = -1
                signals[i] = -0.25
    
    return signals