#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R with 1w Trend Filter
# - Williams %R(14) on 1d for mean-reversion signals: oversold < -80 for long, overbought > -20 for short
# - 1w EMA50 as trend filter: only take long when price > 1w EMA50, short when price < 1w EMA50
# - Williams %R identifies overbought/oversold conditions for mean reversion
# - Weekly EMA filter ensures alignment with higher timeframe trend to avoid counter-trend trades
# - Designed for 1d timeframe with selective entries to avoid overtrading
# - Target: 7-25 trades per year per symbol (30-100 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA50 on 1w timeframe
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Williams %R on 1d timeframe (using daily data from prices)
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    close_1d = prices['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = ((highest_high - close_1d) / (highest_high - lowest_low)) * -100
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in indicators
        if np.isnan(ema_50_1w_aligned[i]) or np.isnan(williams_r[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_1d[i]
        wr = williams_r[i]
        ema50 = ema_50_1w_aligned[i]
        
        if position == 0:
            # Long entry: Williams %R oversold (< -80) + price above 1w EMA50
            if wr < -80 and price > ema50:
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R overbought (> -20) + price below 1w EMA50
            elif wr > -20 and price < ema50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R rises above -50 or price falls below 1w EMA50
            if wr > -50 or price < ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R falls below -50 or price rises above 1w EMA50
            if wr < -50 or price > ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WilliamsR_1wEMA50_TrendFilter"
timeframe = "1d"
leverage = 1.0