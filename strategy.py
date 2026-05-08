#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter combined with 1d RSI mean reversion.
# Long when CHOP(14) > 61.8 (range) AND RSI(14) < 30 on 1d.
# Short when CHOP(14) > 61.8 (range) AND RSI(14) > 70 on 1d.
# Exit when RSI crosses back to neutral (40-60).
# Chop filter avoids trending markets where mean reversion fails.
# RSI extremes provide mean-reversion signals in ranging markets.
# Target: 60-120 total trades over 4 years (15-30/year).

name = "4h_Chop_RSI_MeanReversion"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 4h Choppiness Index: high/low range vs true range over 14 periods
    atr = pd.Series(np.sqrt((high - low)**2)).rolling(window=14, min_periods=14).mean().values
    sum_high_low = pd.Series(high - low).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_high_low / (atr * 14)) / np.log10(14)
    
    # 1d data for RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # RSI on 1d close
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Align 1d RSI to 4h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 14  # Sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if np.isnan(chop[i]) or np.isnan(rsi_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: choppy market (range) + RSI oversold
            long_cond = (chop[i] > 61.8) and (rsi_aligned[i] < 30)
            # Short: choppy market (range) + RSI overbought
            short_cond = (chop[i] > 61.8) and (rsi_aligned[i] > 70)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI returns to neutral (40-60)
            if rsi_aligned[i] >= 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI returns to neutral (40-60)
            if rsi_aligned[i] <= 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals