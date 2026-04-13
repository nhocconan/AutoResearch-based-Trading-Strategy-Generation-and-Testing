#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Keltner breakout with 1d ATR filter and volume confirmation.
# Long: price closes above upper Keltner band + ATR(1d) > median ATR(1d) + volume > 1.5x avg volume
# Short: price closes below lower Keltner band + ATR(1d) > median ATR(1d) + volume > 1.5x avg volume
# Keltner channels: EMA(20) ± 2*ATR(10) on 4h data
# ATR filter ensures trades occur only in volatile regimes, reducing false breakouts in low-volatility periods
# Volume confirmation reduces false breakouts
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Works in both bull and bear markets by filtering for volatility regimes

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h EMA20 for Keltner middle line
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # 4h ATR(10) for Keltner width
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]  # First bar has no previous close
    atr_10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Keltner bands
    upper_keltner = ema_20 + 2 * atr_10
    lower_keltner = ema_20 - 2 * atr_10
    
    # 1-day ATR for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True range for 1d
    tr_1d = np.maximum(high_1d - low_1d, np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]
    atr_1d = pd.Series(tr_1d).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Median ATR(1d) over 50 periods for regime filter
    median_atr_1d = np.full(len(atr_1d), np.nan)
    for i in range(50, len(atr_1d)):
        median_atr_1d[i] = np.median(atr_1d[i-50:i])
    
    # Align 1d ATR and median ATR to 4h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    median_atr_1d_aligned = align_htf_to_ltf(prices, df_1d, median_atr_1d)
    
    # Average volume (20-period = 20*4h = 10 days) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(upper_keltner[i]) or np.isnan(lower_keltner[i]) or 
            np.isnan(atr_1d_aligned[i]) or np.isnan(median_atr_1d_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        atr_1d_val = atr_1d_aligned[i]
        median_atr_1d_val = median_atr_1d_aligned[i]
        upper_k = upper_keltner[i]
        lower_k = lower_keltner[i]
        
        # Volatility filter: current ATR(1d) > median ATR(1d) (high volatility regime)
        vol_filter = atr_1d_val > median_atr_1d_val
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: close above upper Keltner + volatility filter + volume confirmation
            if (price > upper_k and 
                vol_filter and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: close below lower Keltner + volatility filter + volume confirmation
            elif (price < lower_k and 
                  vol_filter and
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes below EMA20 (middle line)
            if price < ema_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price closes above EMA20 (middle line)
            if price > ema_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Keltner_ATR_Volume"
timeframe = "4h"
leverage = 1.0