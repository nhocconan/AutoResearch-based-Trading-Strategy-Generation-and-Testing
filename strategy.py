#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla R3/S3 reversal with 12h EMA34 trend filter and volume confirmation.
Long when price touches Camarilla S3 AND 12h EMA34 rising AND volume > 1.3x 20-period MA.
Short when price touches Camarilla R3 AND 12h EMA34 falling AND volume > 1.3x 20-period MA.
Exit when price reaches Camarilla R1/S1 or 12h EMA34 reverses.
Uses 12h HTF for trend filter to align with higher timeframe direction, volume for momentum confirmation.
Camarilla levels provide precise intraday support/resistance, effective in ranging markets.
Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
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
    
    # Calculate 6h Camarilla levels (based on previous bar's OHLC)
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    camarilla_r1 = np.full(n, np.nan)
    camarilla_s1 = np.full(n, np.nan)
    
    for i in range(1, n):
        # Use previous bar's OHLC to avoid look-ahead
        c = close[i-1]
        h = high[i-1]
        l = low[i-1]
        range_ = h - l
        
        camarilla_r3[i] = c + range_ * 1.1 / 2
        camarilla_s3[i] = c - range_ * 1.1 / 2
        camarilla_r1[i] = c + range_ * 1.1 / 4
        camarilla_s1[i] = c - range_ * 1.1 / 4
    
    # Calculate 12h EMA34 for trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate 6h volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(1, 34, 20)  # Camarilla (needs 1), EMA34, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        r3 = camarilla_r3[i]
        s3 = camarilla_s3[i]
        r1 = camarilla_r1[i]
        s1 = camarilla_s1[i]
        ema_val = ema_34_aligned[i]
        vol_ma_val = vol_ma_20[i]
        
        # Calculate EMA34 slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            ema_prev = ema_34_aligned[i-1]
            ema_rising = ema_val > ema_prev
            ema_falling = ema_val < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        # Volume filter: 6h volume > 1.3x 20-period MA (adaptive to volatility)
        vol_filter = volume[i] > 1.3 * vol_ma_val
        
        if position == 0:
            # Long: Price touches S3 AND EMA34 rising AND volume filter
            if price <= s3 and ema_rising and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Price touches R3 AND EMA34 falling AND volume filter
            elif price >= r3 and ema_falling and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price reaches R1 OR EMA34 starts falling
                if price >= r1 or (i >= start_idx + 1 and ema_val < ema_34_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: price reaches S1 OR EMA34 starts rising
                if price <= s1 or (i >= start_idx + 1 and ema_val > ema_34_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Camarilla_R3S3_Reversal_12hEMA34_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0