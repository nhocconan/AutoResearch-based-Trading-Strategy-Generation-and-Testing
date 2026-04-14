#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w Supertrend for trend filter
    atr_period = 10
    multiplier = 3.0
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR
    atr = np.zeros_like(close)
    for i in range(1, len(tr)):
        if i < atr_period:
            atr[i] = np.mean(tr[:i+1])
        else:
            atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Basic upper and lower bands
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    # Final upper and lower bands
    final_upper = np.zeros_like(close)
    final_lower = np.zeros_like(close)
    supertrend = np.zeros_like(close)
    trend = np.ones_like(close)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(close)):
        if close[i] <= final_upper[i-1]:
            final_upper[i] = upper_band[i]
        else:
            final_upper[i] = upper_band[i]
            
        if close[i] >= final_lower[i-1]:
            final_lower[i] = lower_band[i]
        else:
            final_lower[i] = lower_band[i]
            
        if trend[i-1] == 1:
            if close[i] <= final_lower[i]:
                trend[i] = -1
            else:
                trend[i] = 1
        else:
            if close[i] >= final_upper[i]:
                trend[i] = 1
            else:
                trend[i] = -1
                
        if trend[i] == 1:
            supertrend[i] = final_lower[i]
        else:
            supertrend[i] = final_upper[i]
    
    # Align Supertrend to 1d timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_1w, supertrend)
    trend_aligned = align_htf_to_ltf(prices, df_1w, trend)
    
    # Calculate 1d RSI for momentum filter
    df_1d = get_htf_data(prices, '1d')
    delta = pd.Series(df_1d['close']).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1d = (100 - (100 / (1 + rs))).values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(supertrend_aligned[i]) or 
            np.isnan(trend_aligned[i]) or
            np.isnan(rsi_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Trend filter: Supertrend direction
        trend_filter_long = trend_aligned[i] == 1
        trend_filter_short = trend_aligned[i] == -1
        
        # Momentum filter: RSI between 30 and 70 to avoid extremes
        rsi_filter = (rsi_1d_aligned[i] > 30) & (rsi_1d_aligned[i] < 70)
        
        if position == 0:
            # Long setup: uptrend + momentum filter
            if trend_filter_long and rsi_filter:
                position = 1
                signals[i] = position_size
            # Short setup: downtrend + momentum filter
            elif trend_filter_short and rsi_filter:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: trend changes to downtrend
            if trend_aligned[i] == -1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: trend changes to uptrend
            if trend_aligned[i] == 1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1wSupertrend_RSI_Filter_v1"
timeframe = "1d"
leverage = 1.0