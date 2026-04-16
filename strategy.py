#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R with 1w EMA trend filter
# Long when Williams %R crosses above -80 (oversold) AND price > 1w EMA50
# Short when Williams %R crosses below -20 (overbought) AND price < 1w EMA50
# Exit on opposite Williams %R signal or ATR-based trailing stop (2.5x ATR)
# Williams %R identifies reversals in mean-reverting markets, EMA filter ensures trend alignment
# Target: 20-50 total trades over 4 years (5-12/year) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === 1d Williams %R (14-period) ===
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r[highest_high == lowest_low] = -50  # avoid division by zero
    
    # === 1w EMA50 trend filter ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === 1d ATR for trailing stop (14-period) ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 60
    
    # Track position and exit signals
    position = 0  # 0: flat, 1: long, -1: short
    williams_prev = williams_r[warmup-1] if warmup > 0 else -50
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(williams_r[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        wr = williams_r[i]
        ema_val = ema_50_1w_aligned[i]
        atr_val = atr[i]
        
        # Williams %R signals
        wr_oversold = wr > -80 and williams_prev <= -80  # cross above -80
        wr_overbought = wr < -20 and williams_prev >= -20  # cross below -20
        
        # === TRAILING STOP LOGIC ===
        if position == 1:  # Long position
            # Exit on overbought signal or ATR stop
            if wr_overbought or (atr_val > 0 and price < ema_val - 2.5 * atr_val):
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # Short position
            # Exit on oversold signal or ATR stop
            if wr_oversold or (atr_val > 0 and price > ema_val + 2.5 * atr_val):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Long when: Williams %R oversold cross AND price above 1w EMA50
            if wr_oversold and price > ema_val:
                signals[i] = 0.25
                position = 1
                continue
            # Short when: Williams %R overbought cross AND price below 1w EMA50
            elif wr_overbought and price < ema_val:
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
        
        williams_prev = wr
    
    return signals

name = "1d_WilliamsR14_1wEMA50_ATRTrail_2.5x"
timeframe = "1d"
leverage = 1.0