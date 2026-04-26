#!/usr/bin/env python3
"""
1d_KAMA_Trend_Choppiness_Filter_v2
Hypothesis: Daily KAMA trend with choppiness regime filter and volume confirmation.
- Long when KAMA trending up AND Choppiness Index < 38.2 (trending regime) AND volume > 1.2 * volume_ma(20)
- Short when KAMA trending down AND Choppiness Index < 38.2 (trending regime) AND volume > 1.2 * volume_ma(20)
- Uses 1-week EMA200 as higher timeframe trend filter to avoid counter-trend trades
- Volume confirmation reduces false signals in low participation environments
- Choppiness filter avoids whipsaws in ranging markets (CHOP > 61.8 = range)
- Designed for low frequency (target 7-25 trades/year) to minimize fee drag on 1d timeframe
- Novelty: Combines adaptive trend (KAMA) with regime detection (Choppiness) and volume for BTC/ETH edge in all markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need enough data for calculations
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate KAMA (adaptive trend) - primary signal
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.diff(close)), axis=1)  # sum of |close[t] - close[t-1]| over 10 periods
    # Fix array alignment: change starts at index 10, volatility needs proper calculation
    er = np.full_like(close, np.nan, dtype=float)
    for i in range(10, len(close)):
        if volatility[i] > 0:
            er[i] = change[i] / volatility[i]
        else:
            er[i] = 0
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # for EMA(2)
    slow_sc = 2 / (30 + 1)  # for EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Calculate KAMA
    kama = np.full_like(close, np.nan, dtype=float)
    kama[9] = close[9]  # seed
    for i in range(10, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # KAMA trend: 1 = up (close > KAMA), -1 = down (close < KAMA)
    kama_trend = np.where(close > kama, 1, -1)
    
    # Calculate Choppiness Index (regime filter)
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period has no previous close
    
    # ATR(14) sum
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(atr_sum / (hh - ll)) / log10(14)
    # Avoid division by zero
    range_hl = hh - ll
    chop = np.full_like(close, np.nan, dtype=float)
    mask = (range_hl > 0) & (~np.isnan(atr_sum))
    chop[mask] = 100 * np.log10(atr_sum[mask] / range_hl[mask]) / np.log10(14)
    
    # Regime: CHOP < 38.2 = trending (good for signals), CHOP > 61.8 = ranging (avoid)
    chop_trending = chop < 38.2
    
    # Volume confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.2 * volume_ma)
    
    # Load 1-week data ONCE before loop for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1-week EMA200 for trend filter (needs completed 1w candle)
    ema_200_1w = pd.Series(df_1w['close'].values).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    # 1-week trend: 1 = uptrend (close > EMA200), -1 = downtrend (close < EMA200)
    weekly_trend = np.where(ema_200_1w_aligned > 0, 
                            np.where(close > ema_200_1w_aligned, 1, -1), 
                            0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 200 for EMA, 14 for CHOP, 20 for volume MA)
    start_idx = max(200, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_trend[i]) or np.isnan(chop[i]) or np.isnan(volume_ma[i]) or
            np.isnan(weekly_trend[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Entry conditions: KAMA trend + chop trending regime + volume spike + weekly trend alignment
        if position == 0:
            # Long: KAMA up AND chop trending AND volume spike AND weekly uptrend
            if (kama_trend[i] == 1 and chop_trending[i] and volume_spike[i] and 
                weekly_trend[i] == 1):
                signals[i] = 0.25
                position = 1
            # Short: KAMA down AND chop trending AND volume spike AND weekly downtrend
            elif (kama_trend[i] == -1 and chop_trending[i] and volume_spike[i] and 
                  weekly_trend[i] == -1):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: KAMA turns down OR chop becomes ranging OR weekly trend turns down
            if (kama_trend[i] == -1 or not chop_trending[i] or weekly_trend[i] == -1):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: KAMA turns up OR chop becomes ranging OR weekly trend turns up
            if (kama_trend[i] == 1 or not chop_trending[i] or weekly_trend[i] == 1):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Trend_Choppiness_Filter_v2"
timeframe = "1d"
leverage = 1.0