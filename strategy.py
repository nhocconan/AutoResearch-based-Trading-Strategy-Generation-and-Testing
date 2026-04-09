#!/usr/bin/env python3
# 1d_kama_rsi_chop_regime_v1
# Hypothesis: 1d strategy using KAMA trend direction + RSI extremes + choppiness regime filter.
# Long when KAMA trending up, RSI < 30 (oversold), and choppy market (CHOP > 61.8).
# Short when KAMA trending down, RSI > 70 (overbought), and choppy market (CHOP > 61.8).
# Uses weekly timeframe for regime confirmation only (to avoid look-ahead).
# Discrete position sizing (0.25) to minimize fee churn and manage drawdown.
# Designed to capture mean reversion in choppy markets while avoiding strong trends.
# Target: 7-25 trades/year (30-100 total over 4 years) on BTC/ETH/SOL.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_kama_rsi_chop_regime_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === KAMA Calculation (primary trend) ===
    close_s = pd.Series(close)
    # Efficiency Ratio
    change = abs(close_s - close_s.shift(10))
    volatility = abs(close_s.diff()).rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    kama_trend_up = kama > np.roll(kama, 1)
    kama_trend_up[0] = False
    
    # === RSI Calculation (14-period) ===
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)
    
    # === Choppiness Index (14-period) ===
    atr = np.zeros(n)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(14)
    chop = np.nan_to_num(chop, nan=50.0)
    chop_regime = chop > 61.8  # Choppy market
    
    # === Weekly HTF for regime confirmation (avoid trading against strong weekly trend) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA(21) for trend filter
    close_1w = pd.Series(df_1w['close'].values)
    ema_21_1w = close_1w.ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    weekly_uptrend = close > ema_21_1w_aligned
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(kama_trend_up[i]) or np.isnan(rsi[i]) or 
            np.isnan(chop_regime[i]) or np.isnan(weekly_uptrend[i]) or
            np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI > 50 (mean reversion complete) or weekly trend turns down
            if rsi[i] > 50 or not weekly_uptrend[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI < 50 (mean reversion complete) or weekly trend turns up
            if rsi[i] < 50 or weekly_uptrend[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: KAMA trending up, RSI oversold, choppy market, weekly uptrend
            if (kama_trend_up[i] and rsi[i] < 30 and chop_regime[i] and weekly_uptrend[i]):
                position = 1
                signals[i] = 0.25
            # Enter short: KAMA trending down, RSI overbought, choppy market, weekly downtrend
            elif (~kama_trend_up[i] and rsi[i] > 70 and chop_regime[i] and not weekly_uptrend[i]):
                position = -1
                signals[i] = -0.25
    
    return signals