#!/usr/bin/env python3
"""
6h_ADX_Keltner_MeanReversion_Channel
Hypothesis: Uses ADX to identify low-volatility range-bound markets (ADX < 20) and
trades mean reversion off Keltner Channel boundaries (20-period EMA ± 2*ATR).
In high ADX (>25), follows breakouts in trend direction. Weekly trend filter from 1w
ADX ensures alignment with higher timeframe momentum. Designed for 6h timeframe to
capture multi-day mean reversion and trend continuation with low trade frequency.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Keltner Channel (20-period EMA ± 2*ATR) ===
    # EMA20
    ema20 = np.full(n, np.nan)
    if n >= 20:
        ema20[19] = np.mean(close[0:20])
        alpha = 2 / (20 + 1)
        for i in range(20, n):
            ema20[i] = close[i] * alpha + ema20[i-1] * (1 - alpha)
    
    # True Range and ATR(20)
    tr = np.full(n, np.nan)
    atr = np.full(n, np.nan)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    for i in range(20, n):
        if i == 20:
            atr[i] = np.mean(tr[0:20])
        else:
            atr[i] = (atr[i-1] * 19 + tr[i]) / 20
    
    upper_keltner = ema20 + 2 * atr
    lower_keltner = ema20 - 2 * atr
    
    # === ADX(14) for trend strength ===
    # +DM, -DM
    plus_dm = np.full(n, np.nan)
    minus_dm = np.full(n, np.nan)
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        else:
            plus_dm[i] = 0
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
        else:
            minus_dm[i] = 0
    
    # Smoothed +DM, -DM, TR
    plus_dm_smooth = np.full(n, np.nan)
    minus_dm_smooth = np.full(n, np.nan)
    tr_smooth = np.full(n, np.nan)
    
    for i in range(14, n):
        if i == 14:
            plus_dm_smooth[i] = np.sum(plus_dm[1:15])
            minus_dm_smooth[i] = np.sum(minus_dm[1:15])
            tr_smooth[i] = np.sum(tr[1:15])
        else:
            plus_dm_smooth[i] = plus_dm_smooth[i-1] - (plus_dm_smooth[i-1]/14) + plus_dm[i]
            minus_dm_smooth[i] = minus_dm_smooth[i-1] - (minus_dm_smooth[i-1]/14) + minus_dm[i]
            tr_smooth[i] = tr_smooth[i-1] - (tr_smooth[i-1]/14) + tr[i]
    
    # DI+ and DI-
    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)
    for i in range(14, n):
        if tr_smooth[i] != 0:
            plus_di[i] = 100 * plus_dm_smooth[i] / tr_smooth[i]
            minus_di[i] = 100 * minus_dm_smooth[i] / tr_smooth[i]
    
    # DX and ADX
    dx = np.full(n, np.nan)
    adx = np.full(n, np.nan)
    for i in range(14, n):
        if plus_di[i] + minus_di[i] != 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
    
    for i in range(28, n):
        if i == 28:
            adx[i] = np.mean(dx[14:29])
        else:
            adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # === Weekly trend filter: ADX from 1w ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # ADX(14) on 1w
    tr_1w = np.full(len(close_1w), np.nan)
    atr_1w = np.full(len(close_1w), np.nan)
    tr_1w[0] = high_1w[0] - low_1w[0]
    for i in range(1, len(close_1w)):
        tr_1w[i] = max(high_1w[i] - low_1w[i], abs(high_1w[i] - close_1w[i-1]), abs(low_1w[i] - close_1w[i-1]))
    
    for i in range(14, len(close_1w)):
        if i == 14:
            atr_1w[i] = np.mean(tr_1w[0:14])
        else:
            atr_1w[i] = (atr_1w[i-1] * 13 + tr_1w[i]) / 14
    
    # Simplified ADX calculation for 1w (using same logic as above but abbreviated)
    plus_dm_1w = np.full(len(close_1w), np.nan)
    minus_dm_1w = np.full(len(close_1w), np.nan)
    for i in range(1, len(close_1w)):
        up_move = high_1w[i] - high_1w[i-1]
        down_move = low_1w[i-1] - low_1w[i]
        if up_move > down_move and up_move > 0:
            plus_dm_1w[i] = up_move
        else:
            plus_dm_1w[i] = 0
        if down_move > up_move and down_move > 0:
            minus_dm_1w[i] = down_move
        else:
            minus_dm_1w[i] = 0
    
    plus_dm_smooth_1w = np.full(len(close_1w), np.nan)
    minus_dm_smooth_1w = np.full(len(close_1w), np.nan)
    tr_smooth_1w = np.full(len(close_1w), np.nan)
    
    for i in range(14, len(close_1w)):
        if i == 14:
            plus_dm_smooth_1w[i] = np.sum(plus_dm_1w[1:15])
            minus_dm_smooth_1w[i] = np.sum(minus_dm_1w[1:15])
            tr_smooth_1w[i] = np.sum(tr_1w[1:15])
        else:
            plus_dm_smooth_1w[i] = plus_dm_smooth_1w[i-1] - (plus_dm_smooth_1w[i-1]/14) + plus_dm_1w[i]
            minus_dm_smooth_1w[i] = minus_dm_smooth_1w[i-1] - (minus_dm_smooth_1w[i-1]/14) + minus_dm_1w[i]
            tr_smooth_1w[i] = tr_smooth_1w[i-1] - (tr_smooth_1w[i-1]/14) + tr_1w[i]
    
    plus_di_1w = np.full(len(close_1w), np.nan)
    minus_di_1w = np.full(len(close_1w), np.nan)
    for i in range(14, len(close_1w)):
        if tr_smooth_1w[i] != 0:
            plus_di_1w[i] = 100 * plus_dm_smooth_1w[i] / tr_smooth_1w[i]
            minus_di_1w[i] = 100 * minus_dm_smooth_1w[i] / tr_smooth_1w[i]
    
    dx_1w = np.full(len(close_1w), np.nan)
    adx_1w = np.full(len(close_1w), np.nan)
    for i in range(14, len(close_1w)):
        if plus_di_1w[i] + minus_di_1w[i] != 0:
            dx_1w[i] = 100 * abs(plus_di_1w[i] - minus_di_1w[i]) / (plus_di_1w[i] + minus_di_1w[i])
    
    for i in range(28, len(close_1w)):
        if i == 28:
            adx_1w[i] = np.mean(dx_1w[14:29])
        else:
            adx_1w[i] = (adx_1w[i-1] * 13 + dx_1w[i]) / 14
    
    # Align weekly ADX to 6h
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 28, 28)  # Keltner, ADX, weekly ADX
    
    for i in range(start_idx, n):
        if (np.isnan(upper_keltner[i]) or np.isnan(lower_keltner[i]) or 
            np.isnan(adx[i]) or np.isnan(adx_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime definition
        low_vol_range = adx[i] < 20  # Range-bound market
        strong_trend = adx[i] > 25   # Trending market
        weekly_uptrend = adx_1w_aligned[i] > 25  # Weekly trend up
        weekly_downtrend = adx_1w_aligned[i] < 20  # Weekly trend weak/range
        
        if position == 0:
            # Low volatility: mean reversion at Keltner extremes
            if low_vol_range:
                if close[i] <= lower_keltner[i]:  # Oversold
                    signals[i] = 0.25
                    position = 1
                elif close[i] >= upper_keltner[i]:  # Overbought
                    signals[i] = -0.25
                    position = -1
            # High volatility: breakout in direction of weekly trend
            elif strong_trend:
                if close[i] > upper_keltner[i] and weekly_uptrend:
                    signals[i] = 0.25
                    position = 1
                elif close[i] < lower_keltner[i] and weekly_downtrend:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: return to mean (EMA) or stop if trend reverses
            if close[i] >= ema20[i] or (adx[i] > 30 and close[i] < lower_keltner[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: return to mean or stop if trend reverses
            if close[i] <= ema20[i] or (adx[i] > 30 and close[i] > upper_keltner[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ADX_Keltner_MeanReversion_Channel"
timeframe = "6h"
leverage = 1.0