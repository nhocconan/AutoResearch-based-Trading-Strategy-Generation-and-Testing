#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_AdaptiveKeltner_SR"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Keltner channels and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # 1d ATR for Keltner channels
    atr_period = 14
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr_1d = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # 1d EMA for Keltner center line
    ema_period = 20
    ema_1d = pd.Series(df_1d['close']).ewm(span=ema_period, adjust=False, min_periods=ema_period).mean().values
    
    # Keltner channels: EMA ± 1.5 * ATR
    keltner_upper = ema_1d + 1.5 * atr_1d
    keltner_lower = ema_1d - 1.5 * atr_1d
    
    # 1w EMA for trend filter (long-term bias)
    ema1w_period = 50
    ema_1w = pd.Series(df_1w['close']).ewm(span=ema1w_period, adjust=False, min_periods=ema1w_period).mean().values
    
    # Align all to 6h
    keltner_upper_6h = align_htf_to_ltf(prices, df_1d, keltner_upper)
    keltner_lower_6h = align_htf_to_ltf(prices, df_1d, keltner_lower)
    ema_1d_6h = align_htf_to_ltf(prices, df_1d, ema_1d)
    ema_1w_6h = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(keltner_upper_6h[i]) or np.isnan(keltner_lower_6h[i]) or 
            np.isnan(ema_1d_6h[i]) or np.isnan(ema_1w_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        upper = keltner_upper_6h[i]
        lower = keltner_lower_6h[i]
        ema_1d_val = ema_1d_6h[i]
        ema_1w_val = ema_1w_6h[i]
        
        # Adaptive ATR multiplier based on volatility regime
        # In high volatility, widen bands to avoid whipsaws
        # In low volatility, tighten bands for sensitivity
        atr_ratio = atr_1d[i] / np.mean(atr_1d[max(0, i-20):i+1]) if i >= 20 else 1.0
        adaptive_mult = 1.5 * (0.5 + 0.5 * np.tanh(atr_ratio - 1))  # ranges from ~0.75 to 2.25
        upper_adaptive = ema_1d_val + adaptive_mult * atr_1d[i]
        lower_adaptive = ema_1d_val - adaptive_mult * atr_1d[i]
        
        if position == 0:
            # Long: price breaks above upper band with bullish long-term trend
            if close[i] > upper_adaptive and ema_1w_val > ema_1w_6h[i-1] if i > 0 else True:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band with bearish long-term trend
            elif close[i] < lower_adaptive and ema_1w_val < ema_1w_6h[i-1] if i > 0 else True:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price closes below EMA or re-enters Keltner channel
            if close[i] < ema_1d_val or close[i] < upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price closes above EMA or re-enters Keltner channel
            if close[i] > ema_1d_val or close[i] > lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals