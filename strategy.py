#!/usr/bin/env python3
# 1d_kama_rsi_chop_v4
# Hypothesis: 1d strategy using KAMA trend direction + RSI extremes + chop regime filter.
# KAMA adapts to market noise, reducing whipsaw in ranging markets (2025+).
# RSI < 30 for long, > 70 for short only when chop > 61.8 (strong ranging) to avoid trend-following false signals.
# Discrete sizing (0.0, ±0.25) minimizes fee churn. Target: 30-100 trades over 4 years.
# Primary timeframe: 1d, HTF: 1w for higher-timeframe trend filter (avoid counter-trend in strong weekly trends).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_kama_rsi_chop_v4"
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
    
    # 1w HTF data for trend filter (avoid trading against strong weekly trend)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA50 trend filter
    close_1w = pd.Series(df_1w['close'].values)
    ema_50_1w = close_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # KAMA (1d) - adapts to market efficiency
    close_s = pd.Series(close)
    # Efficiency ratio
    change = abs(close_s.diff(10))
    volatility = close_s.diff().abs().rolling(window=10, min_periods=10).sum()
    er = change / volatility
    er = er.fillna(0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14) - momentum oscillator
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # Choppiness Index (14) - regime filter
    high_14 = pd.Series(high).rolling(window=14, min_periods=14).max()
    low_14 = pd.Series(low).rolling(window=14, min_periods=14).min()
    atr_14 = pd.Series(high - low).rolling(window=14, min_periods=14).sum()
    
    # Avoid division by zero and log10 of zero
    chop_denom = np.log10(atr_14.replace(0, 1e-10)) * np.log10(14)
    chop_denom = chop_denom.replace(0, 1e-10)
    chop = 100 * np.log10((high_14 - low_14) / chop_denom) / np.log10(14)
    chop = chop.fillna(50).values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Weekly trend filter: only trade in direction of weekly EMA50
        weekly_uptrend = close[i] > ema_50_1w_aligned[i]
        weekly_downtrend = close[i] < ema_50_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit: RSI > 50 (momentum fading) or trend change
            if rsi[i] > 50 or not weekly_uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI < 50 (momentum fading) or trend change
            if rsi[i] < 50 or not weekly_downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Only enter when market is ranging (chop > 61.8) to avoid trend-following whipsaw
            if chop[i] > 61.8:
                # Long: RSI oversold + price above KAMA (bullish momentum in range)
                if rsi[i] < 30 and close[i] > kama[i] and weekly_uptrend:
                    position = 1
                    signals[i] = 0.25
                # Short: RSI overbought + price below KAMA (bearish momentum in range)
                elif rsi[i] > 70 and close[i] < kama[i] and weekly_downtrend:
                    position = -1
                    signals[i] = -0.25
    
    return signals