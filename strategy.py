#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume confirmation.
Long when price breaks above Camarilla R3 (12h) AND close > 12h EMA50 AND volume > 1.8x 20-period average.
Short when price breaks below Camarilla S3 (12h) AND close < 12h EMA50 AND volume > 1.8x 20-period average.
Exit when price retraces to Camarilla H4/L4 levels or ATR trailing stop (2.0*ATR from extreme).
Uses discrete position sizing (0.25) and volume filter to target 12-30 trades/year.
6h timeframe reduces noise while capturing medium-term trends in BTC/ETH across bull/bear regimes.
Camarilla levels provide mathematically derived support/resistance proven effective on ETH/SOL.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate 12h Camarilla levels (based on previous 12h bar's OHLC)
    df_12h_prev = df_12h.copy()
    df_12h_prev['high_prev'] = df_12h['high'].shift(1)
    df_12h_prev['low_prev'] = df_12h['low'].shift(1)
    df_12h_prev['close_prev'] = df_12h['close'].shift(1)
    
    # Camarilla calculations: H4/L4 = C_prev ± 1.1*(H_prev-L_prev)/2
    # R3/S3 = C_prev ± 1.1*(H_prev-L_prev)
    high_prev = df_12h_prev['high_prev'].values
    low_prev = df_12h_prev['low_prev'].values
    close_prev = df_12h_prev['close_prev'].values
    rng = high_prev - low_prev
    
    h4 = close_prev + 1.1 * rng / 2.0
    l4 = close_prev - 1.1 * rng / 2.0
    r3 = close_prev + 1.1 * rng
    s3 = close_prev - 1.1 * rng
    
    # Align Camarilla levels to 6h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_12h_prev, h4)
    l4_aligned = align_htf_to_ltf(prices, df_12h_prev, l4)
    r3_aligned = align_htf_to_ltf(prices, df_12h_prev, r3)
    s3_aligned = align_htf_to_ltf(prices, df_12h_prev, s3)
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for trailing stop calculation
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0  # for long trailing stop
    lowest_since_entry = 0.0   # for short trailing stop
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # EMA50 needs 50, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        ema50_val = ema50_12h_aligned[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        h4_val = h4_aligned[i]
        l4_val = l4_aligned[i]
        
        if position == 0:
            # Long: Break above Camarilla R3 AND uptrend (price > EMA50) AND volume spike (1.8x avg)
            if close[i] > r3_val and close[i] > ema50_val and volume[i] > 1.8 * vol_ma_val:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price
            # Short: Break below Camarilla S3 AND downtrend (price < EMA50) AND volume spike (1.8x avg)
            elif close[i] < s3_val and close[i] < ema50_val and volume[i] > 1.8 * vol_ma_val:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = price
        else:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, price)
            elif position == -1:
                lowest_since_entry = min(lowest_since_entry, price)
            
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Price retraces to Camarilla H4/L4 levels
            if position == 1 and close[i] <= h4_val:
                exit_signal = True
            elif position == -1 and close[i] >= l4_val:
                exit_signal = True
            
            # ATR-based trailing stop: 2.0 * ATR from highest/lowest since entry
            if position == 1 and price < highest_since_entry - 2.0 * atr_val:
                exit_signal = True
            elif position == -1 and price > lowest_since_entry + 2.0 * atr_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Camarilla_R3S3_Breakout_12hEMA50_Trend_VolumeConfirmation_H4L4Exit_ATRTrailingStop"
timeframe = "6h"
leverage = 1.0