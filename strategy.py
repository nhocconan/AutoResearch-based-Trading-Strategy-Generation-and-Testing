# 1h_SuperTrend_TripleFilter_v1
# Hypothesis: 1h Supertrend with 4h EMA21 trend filter and 1d RSI momentum filter
# - Supertrend (ATR-based) captures trends while filtering noise
# - 4h EMA21 provides higher timeframe trend bias (avoid counter-trend trades)
# - 1d RSI (40-60) filters extreme momentum to avoid chop whipsaws
# - Designed for low trade frequency (15-30/year) with high win rate
# - Works in bull/bear: follows trend in trending markets, avoids false signals in chop
# - Target: 60-120 total trades over 4 years (15-30/year)

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # === 1. Supertrend on 1h (primary signal) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # ATR calculation
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(alpha=1/10, adjust=False, min_periods=10).mean().values
    
    # Supertrend calculation
    hl2 = (high + low) / 2
    upperband = hl2 + (3.0 * atr)
    lowerband = hl2 - (3.0 * atr)
    
    supertrend = np.zeros(n)
    trend = np.ones(n)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, n):
        if close[i] > upperband[i-1]:
            trend[i] = 1
        elif close[i] < lowerband[i-1]:
            trend[i] = -1
        else:
            trend[i] = trend[i-1]
        
        if trend[i] == 1:
            supertrend[i] = lowerband[i]
        else:
            supertrend[i] = upperband[i]
    
    # === 2. 4h EMA21 trend filter ===
    df_4h = get_htf_data(prices, '4h')
    ema21_4h = pd.Series(df_4h['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema21_4h)
    
    # === 3. 1d RSI momentum filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # === Session filter: 08-20 UTC ===
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if any indicator invalid
        if (np.isnan(supertrend[i]) or np.isnan(ema21_4h_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Check session (08-20 UTC)
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        price = close[i]
        
        if position == 0 and in_session:
            # Long: Supertrend uptrend + price above 4h EMA21 + RSI not extreme
            if (trend[i] == 1 and 
                price > ema21_4h_aligned[i] and 
                40 <= rsi_1d_aligned[i] <= 60):
                signals[i] = 0.20
                position = 1
            
            # Short: Supertrend downtrend + price below 4h EMA21 + RSI not extreme
            elif (trend[i] == -1 and 
                  price < ema21_4h_aligned[i] and 
                  40 <= rsi_1d_aligned[i] <= 60):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: Supertrend downtrend OR outside session
            if trend[i] == -1 or not in_session:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: Supertrend uptrend OR outside session
            if trend[i] == 1 or not in_session:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_SuperTrend_TripleFilter_v1"
timeframe = "1h"
leverage = 1.0