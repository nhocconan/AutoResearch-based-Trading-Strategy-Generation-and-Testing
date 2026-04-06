#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour ATR Breakout with Volume Confirmation and Trend Filter.
# Uses 4h ATR(14) for volatility-based breakout levels (mean ± 2*ATR).
# Volume filter (current volume > 1.5x 20-period average) ensures quality signals.
# Trend filter: EMA(50) direction from 12h timeframe prevents counter-trend entries.
# Works in bull/bear markets via volatility expansion and trend alignment.
# Target: 75-200 total trades over 4 years (19-50/year).

name = "4h_atr_breakout_vol_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ATR(14) calculation
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.full(n, np.nan)
    for i in range(14, n):
        atr[i] = np.nanmean(tr[i-13:i+1])
    
    # Mean price for breakout levels
    mean_price = (high + low) / 2
    
    # Breakout levels: mean ± 2*ATR
    upper_band = mean_price + 2.0 * atr
    lower_band = mean_price - 2.0 * atr
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    # Get 12h EMA(50) for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_12h = np.full(len(close_12h), np.nan)
    for i in range(49, len(close_12h)):
        ema_12h[i] = np.mean(close_12h[i-49:i+1])
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if data not available
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema_12h_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Trend filter: EMA(50) direction
        uptrend = ema_12h_aligned[i] > ema_12h_aligned[i-1]
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price reaches lower band or stoploss
            atr_val = atr[i] if not np.isnan(atr[i]) else 0.001
            stop_loss_level = entry_price - 2.0 * atr_val
            
            if (close[i] <= lower_band[i] or 
                close[i] < stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price reaches upper band or stoploss
            atr_val = atr[i] if not np.isnan(atr[i]) else 0.001
            stop_loss_level = entry_price + 2.0 * atr_val
            
            if (close[i] >= upper_band[i] or 
                close[i] > stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume and trend confirmation
            if volume_filter:
                # Long breakout above upper band in uptrend
                if (close[i] > upper_band[i] and close[i-1] <= upper_band[i] and uptrend):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short breakdown below lower band in downtrend
                elif (close[i] < lower_band[i] and close[i-1] >= lower_band[i] and not uptrend):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals