#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d EMA crossover with 1w trend filter and volume confirmation
# Uses EMA(20) and EMA(50) crossovers on daily timeframe, filtered by weekly EMA(50) trend direction
# and daily volume above 20-period average. Works in bull markets by taking longs when weekly trend up
# and in bear markets by taking shorts when weekly trend down. Position size: 0.25.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d data for EMA and volume ===
    df_1d = get_htf_data(prices, '1d')
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d EMA(20) and EMA(50)
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 1d volume and its 20-period average
    volume_ma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # === 1w data for trend filter ===
    df_1w = get_htf_data(prices, '1w')
    
    close_1w = df_1w['close'].values
    # Weekly EMA(50) for trend direction
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all 1d and 1w data to lower timeframe
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    volume_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma20_1d)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any required data is not available
        if np.isnan(ema20_1d_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or \
           np.isnan(volume_ma20_1d_aligned[i]) or np.isnan(ema50_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Volume filter: current 1d volume > 20-period average volume
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        volume_filter = vol_1d_current > volume_ma20_1d_aligned[i]
        
        # Trend filter: price relative to weekly EMA50
        weekly_uptrend = close_1d[-1] > ema50_1w_aligned[i] if len(close_1d) > 0 else False
        weekly_downtrend = close_1d[-1] < ema50_1w_aligned[i] if len(close_1d) > 0 else False
        
        # EMA crossover signals
        ema_cross_up = ema20_1d_aligned[i] > ema50_1d_aligned[i] and ema20_1d_aligned[i-1] <= ema50_1d_aligned[i-1]
        ema_cross_down = ema20_1d_aligned[i] < ema50_1d_aligned[i] and ema20_1d_aligned[i-1] >= ema50_1d_aligned[i-1]
        
        if position == 0:
            # Long when EMA20 crosses above EMA50 in weekly uptrend with volume
            if ema_cross_up and weekly_uptrend and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short when EMA20 crosses below EMA50 in weekly downtrend with volume
            elif ema_cross_down and weekly_downtrend and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: EMA20 crosses below EMA50 or volume filter fails
            if ema_cross_down or not volume_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: EMA20 crosses above EMA50 or volume filter fails
            if ema_cross_up or not volume_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_EMA_Crossover_1wTrend_VolumeFilter"
timeframe = "1d"
leverage = 1.0