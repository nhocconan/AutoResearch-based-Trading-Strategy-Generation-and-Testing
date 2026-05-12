# 1d_RSI_Momentum_With_Volume_Filter
# Hypothesis: Daily RSI momentum with volume confirmation provides robust trend signals across bull/bear markets.
# Uses RSI(14) > 60 for longs, < 40 for shorts with volume > 1.5x 20-day average.
# Weekly trend filter ensures alignment with higher timeframe momentum.
# Designed for low turnover (<20 trades/year) to minimize fee drag while capturing major moves.
# Works in bull markets via momentum continuation and in bear markets via mean-reversion exhaustion.

#!/usr/bin/env python3
name = "1d_RSI_Momentum_With_Volume_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # RSI calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: current volume > 1.5 * 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    # Weekly trend filter: EMA50 on weekly close
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # ensure EMA50 has enough data
    
    for i in range(start_idx, n):
        # Skip if weekly EMA data not ready
        if np.isnan(ema50_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: RSI > 60 (bullish momentum) + volume expansion + price above weekly EMA50
            if (rsi[i] > 60) and volume_filter[i] and (close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: RSI < 40 (bearish momentum) + volume expansion + price below weekly EMA50
            elif (rsi[i] < 40) and volume_filter[i] and (close[i] < ema50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI falls below 50 (momentum loss) or price crosses below weekly EMA50
            if (rsi[i] < 50) or (close[i] < ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI rises above 50 (momentum loss) or price crosses above weekly EMA50
            if (rsi[i] > 50) or (close[i] > ema50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals