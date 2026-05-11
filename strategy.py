#!/usr/bin/env python3
"""
1d_Weekly_Trend_Daily_Retracement
Hypothesis: In strong weekly uptrend (price above weekly EMA200), buy daily retracements to daily EMA50.
In strong weekly downtrend (price below weekly EMA200), sell short on daily bounces to daily EMA50.
Uses weekly trend filter with daily mean reversion for low-frequency, high-conviction trades.
Designed for <25 trades/year to minimize fee drift and work in both bull/bear markets.
"""

name = "1d_Weekly_Trend_Daily_Retracement"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === WEEKLY DATA FOR TREND FILTER ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Weekly EMA200 for trend filter
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # === DAILY INDICATORS ===
    # Daily EMA50 for entry
    ema50_d = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Daily RSI(14) for overbought/oversold in counter-trend entries
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral when undefined
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers weekly EMA200)
    start_idx = 200
    
    for i in range(start_idx, n):
        # Skip if weekly trend data is invalid
        if np.isnan(ema200_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Weekly uptrend: price above weekly EMA200
            if close[i] > ema200_1w_aligned[i]:
                # Long on daily retracement: price near daily EMA50 with RSI not overbought
                if (abs(close[i] - ema50_d[i]) / ema50_d[i] < 0.02 and  # within 2% of EMA50
                    rsi[i] < 60):  # not overbought
                    signals[i] = 0.25
                    position = 1
            # Weekly downtrend: price below weekly EMA200
            elif close[i] < ema200_1w_aligned[i]:
                # Short on daily bounce: price near daily EMA50 with RSI not oversold
                if (abs(close[i] - ema50_d[i]) / ema50_d[i] < 0.02 and  # within 2% of EMA50
                    rsi[i] > 40):  # not oversold
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: weekly trend turns down OR RSI overbought
            if (close[i] < ema200_1w_aligned[i] or 
                rsi[i] > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: weekly trend turns up OR RSI oversold
            if (close[i] > ema200_1w_aligned[i] or 
                rsi[i] < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals