#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA34 trend filter and volume confirmation.
Long when price breaks above R3 AND 4h EMA34 rising AND volume > 1.5x 20-period MA.
Short when price breaks below S3 AND 4h EMA34 falling AND volume > 1.5x 20-period MA.
Exit when price reverses to opposite Camarilla level (S3 for long, R3 for short) or 4h EMA34 reverses.
Uses 4h HTF for trend filter to avoid counter-trend trades, volume spike for momentum confirmation.
Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.
Camarilla pivots provide precise intraday support/resistance levels effective in ranging markets.
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
    
    # Calculate 1h Camarilla levels (using previous day's OHLC)
    # Camarilla: R4 = close + ((high-low)*1.1/2), R3 = close + ((high-low)*1.1/4)
    #          S3 = close - ((high-low)*1.1/4), S4 = close - ((high-low)*1.1/2)
    # We'll use daily OHLC to calculate Camarilla levels for 1h timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    prev_close = df_1d['close'].shift(1).values  # Previous day close
    prev_high = df_1d['high'].shift(1).values    # Previous day high
    prev_low = df_1d['low'].shift(1).values      # Previous day low
    
    # Calculate Camarilla levels
    rang = prev_high - prev_low
    r3 = prev_close + (rang * 1.1 / 4)
    s3 = prev_close - (rang * 1.1 / 4)
    
    # Align Camarilla levels to 1h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3.values)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3.values)
    
    # Calculate 4h EMA34 for trend filter (HTF)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Calculate 1h volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC (reduces noise trades)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # EMA34, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate EMA34 slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            ema_prev = ema_34_4h_aligned[i-1]
            ema_rising = ema_34_4h_aligned[i] > ema_prev
            ema_falling = ema_34_4h_aligned[i] < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        # Volume filter: 1h volume > 1.5x 20-period MA
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above R3 AND EMA34 rising AND volume filter
            if close[i] > r3_aligned[i] and ema_rising and vol_filter:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S3 AND EMA34 falling AND volume filter
            elif close[i] < s3_aligned[i] and ema_falling and vol_filter:
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price reverses to S3 OR EMA34 starts falling
                if close[i] < s3_aligned[i] or (i >= start_idx + 1 and ema_34_4h_aligned[i] < ema_34_4h_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: price reverses to R3 OR EMA34 starts rising
                if close[i] > r3_aligned[i] or (i >= start_idx + 1 and ema_34_4h_aligned[i] > ema_34_4h_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_Camarilla_R3S3_Breakout_4hEMA34_Trend_VolumeSpike"
timeframe = "1h"
leverage = 1.0