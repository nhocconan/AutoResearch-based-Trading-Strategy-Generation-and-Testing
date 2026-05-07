#!/usr/bin/env python3
name = "6h_Keltner_Channel_Breakout_Volume_1dTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    trend_up = close > ema20_1d_aligned
    trend_down = close < ema20_1d_aligned
    
    # Keltner Channel: EMA(20) +/- ATR(10) * 2
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    tr = np.maximum(high - low, np.absolute(high - np.roll(close, 1)), np.absolute(low - np.roll(close, 1)))
    tr[0] = high[0] - low[0]
    atr10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    upper = ema20 + 2.0 * atr10
    lower = ema20 - 2.0 * atr10
    
    # Volume confirmation: spike > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 10, 20)  # Wait for EMA, ATR, and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(ema20_1d_aligned[i]) or np.isnan(upper[i]) or np.isnan(lower[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close breaks above upper Keltner with volume spike and daily uptrend
            if close[i] > upper[i] and vol_spike[i] and trend_up[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below lower Keltner with volume spike and daily downtrend
            elif close[i] < lower[i] and vol_spike[i] and trend_down[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close below lower Keltner or trend turns down
            if close[i] < lower[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close above upper Keltner or trend turns up
            if close[i] > upper[i] or not trend_down[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Keltner Channel breakouts on 6h timeframe with daily trend filter and volume spike capture strong institutional moves.
# Long when price breaks above upper Keltner (EMA20 + 2*ATR10) with volume confirmation in daily uptrend.
# Short when price breaks below lower Keltner (EMA20 - 2*ATR10) with volume confirmation in daily downtrend.
# Keltner adapts to volatility via ATR, making it effective in both trending and ranging markets.
# Volume spike (>2x average) ensures conviction behind the breakout.
# Designed for 6h timeframe to target 12-37 trades/year, avoiding overtrading.
# Works in bull markets (breaks above upper in uptrend) and bear markets (breaks below lower in downtrend).