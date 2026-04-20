#!/usr/bin/env python3
"""
4h_Hybrid_Trend_Mean_Reversion
Hypothesis: Combine mean reversion (RSI extremes) with trend following (Supertrend) on 4h timeframe.
Long when RSI < 30 and Supertrend up; short when RSI > 70 and Supertrend down.
Use 12h ADX filter to avoid ranging markets. Add volume confirmation for institutional participation.
This hybrid approach captures reversals in trends while avoiding whipsaws in sideways markets.
Target: 80-150 total trades over 4 years (20-37/year) with position size 0.25.
Works in bull/bear: RSI extremes work in both regimes; ADX filter avoids low-quality signals.
"""

name = "4h_Hybrid_Trend_Mean_Reversion"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.full_like(close, np.nan)
    avg_loss = np.full_like(close, np.nan)
    if len(gain) >= 14:
        avg_gain[13] = np.mean(gain[:14])
        avg_loss[13] = np.mean(loss[:14])
        for i in range(14, len(gain)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate Supertrend (ATR=10, multiplier=3)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = np.full_like(close, np.nan)
    if len(tr) >= 10:
        atr[9] = np.mean(tr[:10])
        for i in range(10, len(tr)):
            atr[i] = (atr[i-1] * 9 + tr[i]) / 10
    up = (high + low) / 2 - 3 * atr
    down = (high + low) / 2 + 3 * atr
    supertrend = np.full_like(close, np.nan)
    dir = np.full_like(close, 1)  # 1 for up, -1 for down
    for i in range(1, len(close)):
        if np.isnan(atr[i-1]) or np.isnan(up[i-1]) or np.isnan(down[i-1]):
            supertrend[i] = np.nan
            continue
        if close[i] > supertrend[i-1]:
            dir[i] = 1
        elif close[i] < supertrend[i-1]:
            dir[i] = -1
        else:
            dir[i] = dir[i-1]
        if dir[i] == 1:
            supertrend[i] = max(up[i], supertrend[i-1]) if dir[i-1] == 1 else up[i]
        else:
            supertrend[i] = min(down[i], supertrend[i-1]) if dir[i-1] == -1 else down[i]
    
    # Get 12h ADX for trend strength filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ADX(14) on 12h
    tr_12h = np.maximum(high_12h - low_12h, np.maximum(np.abs(high_12h - np.roll(close_12h, 1)), np.abs(low_12h - np.roll(close_12h, 1))))
    tr_12h[0] = np.nan
    atr_12h = np.full_like(close_12h, np.nan)
    if len(tr_12h) >= 14:
        atr_12h[13] = np.mean(tr_12h[:14])
        for i in range(14, len(tr_12h)):
            atr_12h[i] = (atr_12h[i-1] * 13 + tr_12h[i]) / 14
    
    up_move = np.diff(high_12h)
    down_move = -np.diff(low_12h)
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_di = 100 * np.full_like(close_12h, np.nan)
    minus_di = 100 * np.full_like(close_12h, np.nan)
    if len(atr_12h) >= 14 and not np.all(np.isnan(atr_12h[13:])):
        plus_di_sm = np.full_like(close_12h, np.nan)
        minus_di_sm = np.full_like(close_12h, np.nan)
        if len(plus_dm) >= 14:
            plus_di_sm[13] = np.mean(plus_dm[:14]) * 100 / atr_12h[13] if not np.isnan(atr_12h[13]) and atr_12h[13] != 0 else np.nan
            minus_di_sm[13] = np.mean(minus_dm[:14]) * 100 / atr_12h[13] if not np.isnan(atr_12h[13]) and atr_12h[13] != 0 else np.nan
            for i in range(14, len(close_12h)):
                plus_val = plus_dm[i-1] * 100 / atr_12h[i] if not np.isnan(atr_12h[i]) and atr_12h[i] != 0 else 0
                minus_val = minus_dm[i-1] * 100 / atr_12h[i] if not np.isnan(atr_12h[i]) and atr_12h[i] != 0 else 0
                plus_di_sm[i] = (plus_di_sm[i-1] * 13 + plus_val) / 14
                minus_di_sm[i] = (minus_di_sm[i-1] * 13 + minus_val) / 14
        plus_di = plus_di_sm
        minus_di = minus_di_sm
        dx = np.full_like(close_12h, np.nan)
        divisor = plus_di + minus_di
        mask = (divisor != 0) & ~np.isnan(divisor)
        dx[mask] = 100 * np.abs(plus_di[mask] - minus_di[mask]) / divisor[mask]
        adx_12h = np.full_like(close_12h, np.nan)
        if len(dx) >= 14:
            valid_dx = dx[~np.isnan(dx)]
            if len(valid_dx) >= 14:
                adx_12h[13] = np.mean(valid_dx[:14])
                for i in range(14, len(dx)):
                    if not np.isnan(dx[i]):
                        adx_12h[i] = (adx_12h[i-1] * 13 + dx[i]) / 14
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Volume filter (volume > 1.3x 20-period average)
    vol_ma20 = np.full_like(volume, np.nan)
    for i in range(20, n):
        vol_ma20[i] = np.mean(volume[i-20:i])
    volume_filter = volume > (1.3 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi[i]) or np.isnan(supertrend[i]) or np.isnan(adx_12h_aligned[i]) or 
            np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: RSI < 30 (oversold) + Supertrend up + ADX > 20 + volume spike
            if rsi[i] < 30 and close[i] > supertrend[i] and adx_12h_aligned[i] > 20 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: RSI > 70 (overbought) + Supertrend down + ADX > 20 + volume spike
            elif rsi[i] > 70 and close[i] < supertrend[i] and adx_12h_aligned[i] > 20 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI > 70 (overbought) OR Supertrend flips down
            if rsi[i] > 70 or close[i] < supertrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI < 30 (oversold) OR Supertrend flips up
            if rsi[i] < 30 or close[i] > supertrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals