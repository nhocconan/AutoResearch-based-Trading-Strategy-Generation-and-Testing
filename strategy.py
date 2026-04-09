#!/usr/bin/env python3
# 1d_kama_rsi_chop_regime_v1
# Hypothesis: 1d strategy using KAMA for trend direction, RSI for momentum confirmation, and Choppiness Index for regime filter. Uses 1w HTF EMA(34) for higher timeframe trend alignment. Discrete position sizing (0.25) to minimize fee churn. Target: 7-25 trades/year (30-100 total over 4 years). Works in bull/bear: KAMA adapts to changing market conditions, RSI avoids overextended entries, chop filter identifies trending environments for momentum plays, HTF EMA ensures alignment with weekly trend.

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
    
    # KAMA (adaptive trend)
    close_s = pd.Series(close)
    direction = np.abs(close_s.diff(10).values)
    volatility = np.abs(np.diff(close_s, 1)).values
    volatility = np.concatenate([[0], volatility])  # align length
    er = np.zeros_like(direction)
    for i in range(len(direction)):
        if volatility[i] != 0:
            er[i] = direction[i] / np.sum(volatility[max(0, i-9):i+1])
        else:
            er[i] = 0
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI (14-period momentum)
    delta = np.diff(close_s, 1)
    delta = np.concatenate([[0], delta])  # align length
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Choppiness Index regime filter (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    denominator = np.log10(14) * (highest_high - lowest_low)
    denominator = np.where(denominator == 0, np.nan, denominator)
    chop = 100 * np.log10(atr_sum / denominator)
    
    # Multi-timeframe: 1w EMA(34) trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    close_1w_s = pd.Series(close_1w)
    ema_34_1w = close_1w_s.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or
            np.isnan(close[i]) or np.isnan(ema_34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: chop < 61.8 indicates trending market
        trending_market = chop[i] < 61.8
        # HTF trend filter: price above/below 1w EMA(34)
        htf_uptrend = close[i] > ema_34_1w_aligned[i]
        htf_downtrend = close[i] < ema_34_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit: RSI > 70 (overbought) or price closes below KAMA
            if rsi[i] > 70 or close[i] < kama[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI < 30 (oversold) or price closes above KAMA
            if rsi[i] < 30 or close[i] > kama[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for entry with RSI momentum, regime, and HTF confirmation
            bullish_setup = (close[i] > kama[i]) and (rsi[i] > 50) and (rsi[i] < 70) and trending_market and htf_uptrend
            bearish_setup = (close[i] < kama[i]) and (rsi[i] < 50) and (rsi[i] > 30) and trending_market and htf_downtrend
            
            if bullish_setup:
                position = 1
                signals[i] = 0.25
            elif bearish_setup:
                position = -1
                signals[i] = -0.25
    
    return signals