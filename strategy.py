#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d 10-day EMA trend filter with 1d RSI(14) mean reversion and 1w volume filter
# Takes long when price > 10-day EMA, RSI < 35, and weekly volume > 1.5x weekly average
# Takes short when price < 10-day EMA, RSI > 65, and weekly volume > 1.5x weekly average
# Exits when price crosses back below/above the 10-day EMA
# Designed to capture mean reversion in trending markets with volume confirmation
# Target: 20-40 trades per symbol over 4 years (5-10/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d 10-day EMA
    ema_10 = pd.Series(df_1d['close'].values).ewm(span=10, adjust=False, min_periods=10).values
    
    # Calculate 1d RSI(14)
    delta = np.diff(df_1d['close'].values)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([[np.nan], rsi])
    
    # Calculate 1w volume average (20-period)
    vol_1w = df_1w['volume'].values
    vol_ma_1w = pd.Series(vol_1w).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 1d timeframe
    ema_10_aligned = align_htf_to_ltf(prices, df_1d, ema_10)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 30  # for EMA and RSI calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_10_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(vol_ma_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_1d_current = volume[i] if i < len(volume) else volume[-1]
        
        if position == 0:
            # Long setup: price above EMA, RSI oversold, weekly volume spike
            if (price > ema_10_aligned[i] and 
                rsi_aligned[i] < 35 and 
                vol_1d_current > 1.5 * vol_ma_1w_aligned[i]):
                position = 1
                signals[i] = position_size
            # Short setup: price below EMA, RSI overbought, weekly volume spike
            elif (price < ema_10_aligned[i] and 
                  rsi_aligned[i] > 65 and 
                  vol_1d_current > 1.5 * vol_ma_1w_aligned[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below EMA
            if price < ema_10_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above EMA
            if price > ema_10_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_EMA10_RSI_VolumeFilter"
timeframe = "1d"
leverage = 1.0