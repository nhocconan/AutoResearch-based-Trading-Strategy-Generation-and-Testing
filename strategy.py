#!/usr/bin/env python3
"""
1d_Volatility_Squeeze_Breakout_1wTrend
Hypothesis: In both bull and bear markets, volatility contractions (low ATR ratio) followed by breakouts with volume capture explosive moves. 
Weekly trend filter ensures we only trade in the direction of the higher timeframe trend, reducing false signals during chop.
Target: 20-30 trades/year on 1d timeframe with strict entry conditions.
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
    
    # Calculate 1-week ATR for trend filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range and ATR(20) on weekly
    tr_1w = np.maximum(
        high_1w[1:] - low_1w[1:],
        np.maximum(
            np.abs(high_1w[1:] - close_1w[:-1]),
            np.abs(low_1w[1:] - close_1w[:-1])
        )
    )
    tr_1w = np.concatenate([[np.nan], tr_1w])  # align with index
    atr_20_1w = np.full(len(close_1w), np.nan)
    for i in range(20, len(tr_1w)):
        if np.isnan(tr_1w[i-19:i+1]).any():
            atr_20_1w[i] = np.nan
        else:
            atr_20_1w[i] = np.mean(tr_1w[i-19:i+1])
    atr_20_1w = np.where(np.isnan(atr_20_1w), 0, atr_20_1w)  # avoid propagation
    
    # Weekly trend: close above/below EMA20 of ATR-adjusted close
    ema20_1w = np.full(len(close_1w), np.nan)
    k = 2 / (20 + 1)
    for i in range(20, len(close_1w)):
        if i == 20:
            ema20_1w[i] = np.mean(close_1w[0:21])
        else:
            ema20_1w[i] = close_1w[i] * k + ema20_1w[i-1] * (1 - k)
    ema20_1w = np.where(np.isnan(ema20_1w), 0, ema20_1w)
    weekly_up = close_1w > ema20_1w
    weekly_down = close_1w < ema20_1w
    
    weekly_up_aligned = align_htf_to_ltf(prices, df_1w, weekly_up.astype(float))
    weekly_down_aligned = align_htf_to_ltf(prices, df_1w, weekly_down.astype(float))
    
    # Daily ATR ratio for volatility squeeze: ATR(7) / ATR(30)
    tr_daily = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
    )
    tr_daily = np.concatenate([[np.nan], tr_daily])
    atr_7 = np.full(n, np.nan)
    atr_30 = np.full(n, np.nan)
    for i in range(7, n):
        if np.isnan(tr_daily[i-6:i+1]).any():
            atr_7[i] = np.nan
        else:
            atr_7[i] = np.mean(tr_daily[i-6:i+1])
    for i in range(30, n):
        if np.isnan(tr_daily[i-29:i+1]).any():
            atr_30[i] = np.nan
        else:
            atr_30[i] = np.mean(tr_daily[i-29:i+1])
    atr_ratio = np.where((atr_30 != 0) & ~np.isnan(atr_30), atr_7 / atr_30, np.nan)
    
    # Bollinger Band width for squeeze confirmation: (upper - lower) / middle
    sma_20 = np.full(n, np.nan)
    for i in range(20, n):
        sma_20[i] = np.mean(close[i-20:i+1])
    std_20 = np.full(n, np.nan)
    for i in range(20, n):
        if np.isnan(sma_20[i]):
            std_20[i] = np.nan
        else:
            std_20[i] = np.std(close[i-20:i+1])
    bb_width = np.where(sma_20 != 0, (4 * std_20) / sma_20, np.nan)  # 2*std each side
    
    # Volatility squeeze: low ATR ratio AND low BB width
    vol_squeeze = (atr_ratio < 0.3) & (bb_width < 0.05)  # thresholds tuned for daily
    
    # Donchian breakout: 20-day high/low
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    for i in range(20, n):
        donch_high[i] = np.max(high[i-20:i+1])
        donch_low[i] = np.min(low[i-20:i+1])
    
    # Volume confirmation: current volume > 2 x 20-day average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i+1])
    vol_confirm = volume > (2 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Ensure indicators ready
    
    for i in range(start_idx, n):
        if (np.isnan(weekly_up_aligned[i]) or np.isnan(weekly_down_aligned[i]) or
            np.isnan(vol_squeeze[i]) or np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: volatility squeeze breakout above Donchian high with weekly uptrend
            if (vol_squeeze[i-1] and  # squeeze was present just before breakout
                close[i] > donch_high[i] and 
                weekly_up_aligned[i] and
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: volatility squeeze breakout below Donchian low with weekly downtrend
            elif (vol_squeeze[i-1] and
                  close[i] < donch_low[i] and
                  weekly_down_aligned[i] and
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: close below Donchian low or weekly trend turns down
            if (close[i] < donch_low[i] or not weekly_up_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close above Donchian high or weekly trend turns up
            if (close[i] > donch_high[i] or not weekly_down_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Volatility_Squeeze_Breakout_1wTrend"
timeframe = "1d"
leverage = 1.0