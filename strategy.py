#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Bollinger Band Width regime filter + RSI mean reversion + Weekly trend confirmation
# Long when BBW < 20th percentile (squeeze), RSI < 30 (oversold), weekly uptrend
# Short when BBW < 20th percentile (squeeze), RSI > 70 (overbought), weekly downtrend
# Bollinger Band Width identifies low volatility periods conducive to mean reversion
# RSI captures oversold/overbought conditions within the squeeze
# Weekly trend ensures trades align with higher timeframe momentum
# Targets 30-100 total trades over 4 years (7-25/year) to minimize fee drag

name = "1d_BBW_RSI_WeeklyTrend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA(34) for trend filter
    weekly_close = df_1w['close'].values
    ema34_1w = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper = sma + bb_std * std
    lower = sma - bb_std * std
    bbw = (upper - lower) / sma  # Bollinger Band Width
    
    # BBW percentile rank (20-period lookback)
    bbw_rank = np.zeros_like(bbw)
    for i in range(bb_period, n):
        start = max(0, i - 20)
        window = bbw[start:i+1]
        if len(window) > 0:
            bbw_rank[i] = (np.sum(window <= bbw[i]) / len(window)) * 100
        else:
            bbw_rank[i] = 50
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for calculations
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(bbw_rank[i]) or np.isnan(rsi[i]) or 
            np.isnan(ema34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        bbw_rank_val = bbw_rank[i]
        rsi_val = rsi[i]
        ema34_1w_val = ema34_1w_aligned[i]
        
        if position == 0:
            # Enter long: BBW squeeze (<20th percentile), RSI oversold (<30), weekly uptrend
            if bbw_rank_val < 20 and rsi_val < 30 and ema34_1w_val > 0:
                signals[i] = 0.25
                position = 1
            # Enter short: BBW squeeze (<20th percentile), RSI overbought (>70), weekly downtrend
            elif bbw_rank_val < 20 and rsi_val > 70 and ema34_1w_val < 0:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: BBW expansion (>50th percentile) or RSI overbought (>70) or weekly trend down
            if bbw_rank_val > 50 or rsi_val > 70 or ema34_1w_val < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: BBW expansion (>50th percentile) or RSI oversold (<30) or weekly trend up
            if bbw_rank_val > 50 or rsi_val < 30 or ema34_1w_val > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals