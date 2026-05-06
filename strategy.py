#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA direction filter with 1w EMA34 trend and volume confirmation
# Long when KAMA(14,2,30) > KAMA_prev AND 1w close > 1w EMA34 AND volume > 1.5 * 20-bar average volume
# Short when KAMA < KAMA_prev AND 1w close < 1w EMA34 AND volume > 1.5 * 20-bar average volume
# Exit when KAMA reverses direction (KAMA crosses below/above previous KAMA)
# Uses discrete sizing 0.25 to balance return and fee drag
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe
# KAMA adapts to market noise, reducing whipsaws in ranging markets while capturing trends
# 1w EMA34 provides strong multi-timeframe trend filter for better regime adaptation
# Volume confirmation reduces false signals during low participation periods

name = "1d_KAMA_1wEMA34_Volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) for 1d timeframe
    # ER = Efficiency Ratio, Smooth = smoothing constant
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if False else None  # placeholder for correct calc
    
    # Proper KAMA calculation
    close_series = pd.Series(close)
    change = close_series.diff().abs()
    volatility = close_series.diff().abs().rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, np.nan)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2  # fast=2, slow=30
    sc = sc.fillna(0)  # handle NaN from division by zero
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    # Get 1w data ONCE before loop for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA34
    close_1w_series = pd.Series(close_1w)
    ema34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 1d timeframe (wait for completed HTF bar)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume confirmation: volume > 1.5 * 20-bar average volume
    volume_series = pd.Series(volume)
    avg_volume_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirmation = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    prev_kama = kama[0]  # initialize previous KAMA
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(kama[i]) or np.isnan(ema34_1w_aligned[i]) or np.isnan(volume_confirmation[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            prev_kama = kama[i] if not np.isnan(kama[i]) else prev_kama
            continue
        
        if position == 0:
            # KAMA direction signals with trend and volume filters
            # Long: KAMA rising AND uptrend AND volume confirmation
            if kama[i] > prev_kama and close[i] > ema34_1w_aligned[i] and volume_confirmation[i]:
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling AND downtrend AND volume confirmation
            elif kama[i] < prev_kama and close[i] < ema34_1w_aligned[i] and volume_confirmation[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: KAMA reverses (falls below previous KAMA)
            if kama[i] < prev_kama:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: KAMA reverses (rises above previous KAMA)
            if kama[i] > prev_kama:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        
        prev_kama = kama[i]
    
    return signals