#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla pivot breakout with 12h EMA50 trend filter and volume spike confirmation.
Long when price breaks above Camarilla R3 AND 12h EMA50 rising AND 6h volume > 2.0x 20-period MA.
Short when price breaks below Camarilla S3 AND 12h EMA50 falling AND 6h volume > 2.0x 20-period MA.
Exit when price touches opposite Camarilla level (R2/S2) or 12h EMA50 reverses.
Uses 12h HTF for trend filter to avoid counter-trend trades, volume spike for momentum confirmation.
Target: 75-150 total trades over 4 years (19-37/year) for 6h timeframe.
Camarilla levels provide intraday structure, 12h EMA50 filters major trend, volume spike avoids low-momentum breakouts.
Works in bull (trend filters) and bear (volume spikes on breakdowns).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 6h Camarilla levels (based on previous bar's range)
    camarilla_r3 = np.full(n, np.nan)
    camarilla_r2 = np.full(n, np.nan)
    camarilla_s2 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    
    for i in range(1, n):
        prev_close = close[i-1]
        prev_high = high[i-1]
        prev_low = low[i-1]
        range_val = prev_high - prev_low
        
        camarilla_r3[i] = prev_close + range_val * 1.1 / 2
        camarilla_r2[i] = prev_close + range_val * 1.1 / 4
        camarilla_s2[i] = prev_close - range_val * 1.1 / 4
        camarilla_s3[i] = prev_close - range_val * 1.1 / 2
    
    # Calculate 12h EMA50 for trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 6h volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(1, 50, 20)  # Camarilla, EMA50, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_r2[i]) or 
            np.isnan(camarilla_s2[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        r3 = camarilla_r3[i]
        r2 = camarilla_r2[i]
        s2 = camarilla_s2[i]
        s3 = camarilla_s3[i]
        ema_val = ema_50_aligned[i]
        vol_ma_val = vol_ma_20[i]
        
        # Calculate EMA50 slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            ema_prev = ema_50_aligned[i-1]
            ema_rising = ema_val > ema_prev
            ema_falling = ema_val < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        # Volume filter: 6h volume > 2.0x 20-period MA (adaptive to volatility)
        vol_filter = volume[i] > 2.0 * vol_ma_val
        
        if position == 0:
            # Long: Break above Camarilla R3 AND EMA50 rising AND volume filter
            if price > r3 and ema_rising and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Break below Camarilla S3 AND EMA50 falling AND volume filter
            elif price < s3 and ema_falling and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price touches Camarilla R2 OR EMA50 starts falling
                if price < r2 or (i >= start_idx + 1 and ema_val < ema_50_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: price touches Camarilla S2 OR EMA50 starts rising
                if price > s2 or (i >= start_idx + 1 and ema_val > ema_50_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Camarilla_R3S3_Breakout_12hEMA50_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0