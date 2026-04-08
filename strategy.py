#!/usr/bin/env python3
# 1d_1w_kama_rsi_chop_v1
# Hypothesis: Trade KAMA direction on 1d with RSI mean reversion and Choppiness filter on 1w.
# In choppy markets (high Choppiness), buy RSI oversold when KAMA turns up, sell RSI overbought when KAMA turns down.
# Avoids strong trends where mean reversion fails. Uses RSI extremes for entries and KAMA for trend confirmation.
# Target: 15-25 trades/year with strict entry conditions to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_kama_rsi_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1w Choppiness Index (14-period)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # +DI and -DI for ADX
    up_move = np.diff(high_1w, prepend=high_1w[0])
    down_move = np.diff(low_1w, prepend=low_1w[0]) * -1  # invert so down move is positive
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed +DM, -DM, TR
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values
    tr_smooth = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    
    # +DI and -DI
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # Choppiness Index
    chop = 100 * np.log10(atr_1w.sum() / (pd.Series(atr_1w).rolling(window=14, min_periods=14).sum().values)) / np.log10(14)
    chop = np.where(tr_smooth > 0, chop, 50.0)  # fallback to 50 when no movement
    
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    # 1d KAMA (close, ER=10, FAST=2, SLOW=30)
    change = np.abs(np.diff(close, k=10, prepend=close[:10]))
    volatility = np.sum(np.abs(np.diff(close, k=1, prepend=close[0])), axis=0)
    er = np.where(volatility > 0, change / volatility, 0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # 1d RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(chop_aligned[i]) or np.isnan(kama[i]) or np.isnan(rsi[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Chop threshold: chop > 61.8 = ranging market (good for mean reversion)
        ranging = chop_aligned[i] > 61.8
        
        if position == 1:  # Long position
            # Exit: RSI overbought OR KAMA turns down
            if rsi[i] > 70 or kama[i] < kama[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI oversold OR KAMA turns up
            if rsi[i] < 30 or kama[i] > kama[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only trade in ranging markets
            if ranging:
                # Long entry: RSI oversold and KAMA turning up
                if rsi[i] < 30 and kama[i] > kama[i-1]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: RSI overbought and KAMA turning down
                elif rsi[i] > 70 and kama[i] < kama[i-1]:
                    position = -1
                    signals[i] = -0.25
    
    return signals