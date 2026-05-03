#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA50 trend filter and ATR volatility filter
# Elder Ray measures bull/bear power relative to EMA13 to identify trend strength
# 1d EMA50 ensures alignment with daily trend to avoid counter-trend trades
# ATR(14) filter ensures sufficient volatility for meaningful moves
# Target: 80-120 total trades over 4 years (20-30/year) to balance edge and fee drag
# Works in both bull and bear markets by following the higher timeframe trend

name = "6h_ElderRay_1dEMA50_ATRFilter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 6h EMA(13) for Elder Ray
    close_s = pd.Series(close)
    ema_13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Calculate ATR(14) for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_s = pd.Series(tr)
    atr_14 = atr_s.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start from 50 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr_14[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # ATR filter: require ATR > 0.5% of price to ensure sufficient volatility
        atr_filter = atr_14[i] > (0.005 * close[i])
        
        # Elder Ray signals with 1d trend filter
        # Long: Bull Power > 0 (strong buying) + price above 1d EMA50 + ATR filter
        # Short: Bear Power < 0 (strong selling) + price below 1d EMA50 + ATR filter
        if position == 0:
            if bull_power[i] > 0 and close[i] > ema_50_1d_aligned[i] and atr_filter:
                signals[i] = 0.25
                position = 1
            elif bear_power[i] < 0 and close[i] < ema_50_1d_aligned[i] and atr_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power turns negative (weakening buying) OR price below 1d EMA50
            if bull_power[i] <= 0 or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power turns positive (weakening selling) OR price above 1d EMA50
            if bear_power[i] >= 0 or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals