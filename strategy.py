#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d KAMA + RSI + Chop regime filter
# - KAMA identifies adaptive trend direction
# - RSI(14) provides overbought/oversold signals with trend filter
# - Choppiness Index (CHOP) filters regimes: CHOP > 61.8 = range (mean revert), CHOP < 38.2 = trending (trend follow)
# - Long: KAMA up AND RSI < 40 AND CHOP > 61.8 (oversold in range)
# - Short: KAMA down AND RSI > 60 AND CHOP > 61.8 (overbought in range)
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 7-25 trades/year (30-100 total over 4 years) to stay within fee drag limits
# - Works in both bull (trend following via KAMA) and bear (mean reversion in range via RSI+CHOP)

name = "1d_kama_rsi_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1w data ONCE before loop for regime context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return signals
    
    # Pre-compute 1w KAMA for trend filter
    close_1w = df_1w['close'].values
    # Calculate Efficiency Ratio (ER) for KAMA
    change = np.abs(np.diff(close_1w, n=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close_1w, n=1)), axis=0)  # 10-period volatility
    # Handle edge cases for volatility calculation
    volatility_padded = np.concatenate([np.full(9, np.nan), volatility])
    er = np.where(volatility_padded != 0, change / volatility_padded, 0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Calculate KAMA
    kama_1w = np.full_like(close_1w, np.nan)
    kama_1w[9] = close_1w[9]  # Start after first 10 bars
    for i in range(10, len(close_1w)):
        if np.isnan(kama_1w[i-1]):
            kama_1w[i] = close_1w[i]
        else:
            kama_1w[i] = kama_1w[i-1] + sc[i] * (close_1w[i] - kama_1w[i-1])
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # Pre-compute 1d RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Pre-compute 1d Choppiness Index (CHOP)
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Set first TR to high-low
    tr[0] = tr1[0]
    # Sum of TR over 14 periods
    tr_sum_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # CHOP formula: 100 * log10(sum_TR / (HH - LL)) / log10(14)
    chop = 100 * np.log10(tr_sum_14 / (hh_14 - ll_14 + 1e-10)) / np.log10(14)
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(kama_1w_aligned[i]) or np.isnan(rsi[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # KAMA trend direction (using 1w KAMA slope)
        kama_up = kama_1w_aligned[i] > kama_1w_aligned[i-1]
        kama_down = kama_1w_aligned[i] < kama_1w_aligned[i-1]
        
        # RSI conditions
        rsi_oversold = rsi[i] < 40
        rsi_overbought = rsi[i] > 60
        
        # Chop regime: CHOP > 61.8 = ranging market (good for mean reversion)
        chop_range = chop[i] > 61.8
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: KAMA up AND RSI oversold AND choppy/ranging market
        if kama_up and rsi_oversold and chop_range:
            enter_long = True
        
        # Short: KAMA down AND RSI overbought AND choppy/ranging market
        if kama_down and rsi_overbought and chop_range:
            enter_short = True
        
        # Exit conditions: reverse signals or regime change
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if KAMA turns down OR RSI becomes overbought OR chop regime ends
            exit_long = (not kama_up) or (rsi[i] > 70) or (chop[i] < 50)
        elif position == -1:
            # Exit short if KAMA turns up OR RSI becomes oversold OR chop regime ends
            exit_short = (not kama_down) or (rsi[i] < 30) or (chop[i] < 50)
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals