#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla R3/S3 Reversal with 1d EMA34 trend filter and volume confirmation.
Long when price rejects S3 (bullish candle close > S3) AND 1d EMA34 rising AND volume > 1.3x 20-period MA.
Short when price rejects R3 (bearish candle close < R3) AND 1d EMA34 falling AND volume > 1.3x 20-period MA.
Exit when price touches opposite Camarilla level (S4 for longs, R4 for shorts) or 1d EMA34 reverses.
Uses 1d HTF for trend filter to avoid counter-trend trades, volume confirmation for momentum.
Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
Camarilla provides mean-reversion levels, 1d EMA34 filters major trend, volume confirms rejection strength.
Works in both bull and bear markets by following higher timeframe trend and fading extremes.
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
    camarilla_r4 = np.full(n, np.nan)
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    camarilla_s4 = np.full(n, np.nan)
    
    for i in range(1, n):
        # Use previous bar's high, low, close (avoid look-ahead)
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        range_val = prev_high - prev_low
        
        camarilla_r4[i] = prev_close + range_val * 1.1 / 2
        camarilla_r3[i] = prev_close + range_val * 1.1 / 4
        camarilla_s3[i] = prev_close - range_val * 1.1 / 4
        camarilla_s4[i] = prev_close - range_val * 1.1 / 2
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 6h volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(1, 34, 20)  # Camarilla (needs 1), EMA34, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(camarilla_r4[i]) or np.isnan(camarilla_s4[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        r3 = camarilla_r3[i]
        s3 = camarilla_s3[i]
        r4 = camarilla_r4[i]
        s4 = camarilla_s4[i]
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
        
        # Bullish rejection: close above S3 (price found support at S3)
        bullish_rejection = price > s3
        # Bearish rejection: close below R3 (price found resistance at R3)
        bearish_rejection = price < r3
        
        if position == 0:
            # Long: Bullish rejection at S3 AND EMA34 rising AND volume filter
            if bullish_rejection and ema_rising and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Bearish rejection at R3 AND EMA34 falling AND volume filter
            elif bearish_rejection and ema_falling and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price touches S4 (stop) OR EMA34 starts falling
                if price <= s4 or (i >= start_idx + 1 and ema_val < ema_34_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: price touches R4 (stop) OR EMA34 starts rising
                if price >= r4 or (i >= start_idx + 1 and ema_val > ema_34_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Camarilla_R3S3_Reversal_1dEMA34_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0