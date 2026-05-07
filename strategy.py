#!/usr/bin/env python3
name = "1d_KAMA_Trend_RSI_Filter_v2"
timeframe = "1d"
leverage = 1.0

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
    
    # Load weekly data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # KAMA: Kaufman Adaptive Moving Average
    # ER = Efficiency Ratio
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close))
    er = np.zeros_like(change)
    for i in range(len(change)):
        if volatility[i] != 0:
            er[i] = change[i] / volatility[i]
        else:
            er[i] = 0
    # Smooth ER
    er_smooth = pd.Series(er).ewm(alpha=0.1, adjust=False).fillna(0).values
    # Smoothing constants
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er_smooth * (fast_sc - slow_sc) + slow_sc) ** 2
    # KAMA calculation
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Weekly EMA for trend filter
    ema_30_1w = pd.Series(df_1w['close']).ewm(span=30, adjust=False, min_periods=30).mean().values
    ema_30_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_30_1w)
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume spike detection
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(ema_30_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # KAMA slope (trend direction)
        kama_slope = kama[i] - kama[i-1]
        
        # Volume condition
        vol_condition = volume[i] > vol_ma_20[i] * 1.5
        
        if position == 0:
            # Long: price above KAMA, RSI > 50, weekly uptrend, volume spike
            if close[i] > kama[i] and rsi[i] > 50 and ema_30_1w_aligned[i] > ema_30_1w_aligned[i-1] and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA, RSI < 50, weekly downtrend, volume spike
            elif close[i] < kama[i] and rsi[i] < 50 and ema_30_1w_aligned[i] < ema_30_1w_aligned[i-1] and vol_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price below KAMA or RSI < 40
            if close[i] < kama[i] or rsi[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price above KAMA or RSI > 60
            if close[i] > kama[i] or rsi[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: KAMA adapts to market noise - reduces whipsaws in sideways markets
# - KAMA follows price closely in trends, flattens in ranges (reduces false signals)
# - Entry: price crosses KAMA with RSI confirmation and weekly trend alignment
# - Volume spike (1.5x average) confirms institutional participation
# - Weekly EMA30 filter ensures alignment with higher timeframe trend
# - Exit when price returns to KAMA or RSI reaches extreme levels
# - Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend)
# - Position size 0.25 targets ~20-50 trades/year to minimize fee drag
# - KAMA's adaptive nature reduces whipsaws vs fixed MA in choppy markets
# - RSI filter prevents overextended entries
# - Weekly trend filter avoids counter-trend trades
# - Simple, robust logic with clear entry/exit conditions