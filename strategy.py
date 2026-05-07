#!/usr/bin/env python3
name = "12h_KAMA_Trend_RSI_MeanReversion"
timeframe = "12h"
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
    
    # Load daily data ONCE for trend filter and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # KAMA on 12h close
    change = np.abs(np.diff(close, prepend=close[0]))
    direction = np.abs(np.diff(close, n=10, prefill=close[:10]))
    er = np.where(direction != 0, change / direction, 0)
    sc = (er * (0.6667 - 0.0645) + 0.0645) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Daily RSI for mean reversion
    delta = np.diff(df_1d['close'], prepend=df_1d['close'][0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi_1d = rsi
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Daily EMA for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    
    start_idx = max(50, 14)
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Price above KAMA = uptrend, below = downtrend
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # RSI conditions for mean reversion
        rsi_oversold = rsi_1d_aligned[i] < 30
        rsi_overbought = rsi_1d_aligned[i] > 70
        
        # Trend filter: price relative to daily EMA50
        price_above_ema = close[i] > ema_50_1d_aligned[i]
        price_below_ema = close[i] < ema_50_1d_aligned[i]
        
        # Entry conditions
        if price_above_kama and rsi_oversold and price_above_ema:
            # Long: uptrend + oversold + above daily EMA
            signals[i] = 0.25
        elif price_below_kama and rsi_overbought and price_below_ema:
            # Short: downtrend + overbought + below daily EMA
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

# Hypothesis: 12h KAMA trend filter with daily RSI mean reversion
# - KAMA adapts to market noise, providing smooth trend identification
# - In uptrend (price > KAMA), look for long when daily RSI is oversold (<30)
# - In downtrend (price < KAMA), look for short when daily RSI is overbought (>70)
# - Daily EMA50 filter ensures alignment with higher timeframe trend
# - Works in both bull (buy dips in uptrend) and bear (sell rallies in downtrend)
# - Mean reversion component reduces whipsaws vs pure trend following
# - Position size 0.25 targets ~15-35 trades/year to avoid fee drag
# - Combines adaptive trend (KAMA) with oscillator extreme (RSI) for confluence
# - Aims for 60-140 total trades over 4 years (15-35/year) to stay within limits
# - Uses daily timeframe for RSI and EMA to reduce noise vs lower timeframes
# - Avoids overtrading by requiring multiple conditions to align simultaneously