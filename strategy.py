#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h EMA crossover with 1d RSI filter and volume confirmation
# Takes long when 6h EMA(9) crosses above EMA(21) with 1d RSI > 50 and volume spike
# Takes short when 6h EMA(9) crosses below EMA(21) with 1d RSI < 50 and volume spike
# Exits when EMA crossover reverses or volume drops below average
# Designed to capture momentum with trend confirmation from higher timeframe
# Target: 50-150 total trades over 4 years (12-37/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 6h EMA(9) and EMA(21)
    ema9 = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate 1d RSI(14)
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Calculate 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 6h timeframe
    ema9_aligned = ema9  # already on 6h
    ema21_aligned = ema21  # already on 6h
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50  # for EMA and RSI calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema9_aligned[i]) or np.isnan(ema21_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_1d_current = vol_1d[i] if i < len(vol_1d) else vol_1d[-1]
        
        # EMA crossover signals
        ema_cross_up = ema9_aligned[i] > ema21_aligned[i] and ema9_aligned[i-1] <= ema21_aligned[i-1]
        ema_cross_down = ema9_aligned[i] < ema21_aligned[i] and ema9_aligned[i-1] >= ema21_aligned[i-1]
        
        if position == 0:
            # Long setup: EMA bullish cross with RSI > 50 and volume spike
            if (ema_cross_up and 
                rsi_1d_aligned[i] > 50 and 
                vol_1d_current > 1.5 * vol_ma_1d_aligned[i]):
                position = 1
                signals[i] = position_size
            # Short setup: EMA bearish cross with RSI < 50 and volume spike
            elif (ema_cross_down and 
                  rsi_1d_aligned[i] < 50 and 
                  vol_1d_current > 1.5 * vol_ma_1d_aligned[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: EMA bearish cross or volume drops
            if ema_cross_down or vol_1d_current < vol_ma_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: EMA bullish cross or volume drops
            if ema_cross_up or vol_1d_current < vol_ma_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_EMA_Crossover_1dRSI_Volume"
timeframe = "6h"
leverage = 1.0