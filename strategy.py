#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA(50) trend filter and volume confirmation
# Donchian breakout captures institutional participation in trending markets
# 12h EMA(50) ensures alignment with higher timeframe trend to avoid range whipsaws
# Volume confirmation filters false breakouts
# Designed for 4h timeframe targeting 19-50 trades/year (75-200 total over 4 years)
# Uses discrete position sizing (0.25) to minimize fee churn and control drawdown
# Works in bull markets (breakout above upper band + 12h EMA up) and bear markets (breakdown below lower band + 12h EMA down)

name = "4h_Donchian20_12hEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h data for trend filter (EMA) 
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # Need enough for EMA calculation
        return np.zeros(n)
    
    # 12h EMA(50)
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Donchian(20) on primary timeframe (4h)
    # Upper band = highest high of last 20 periods
    # Lower band = lowest low of last 20 periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmation = volume > (1.5 * vol_ema_20)  # Volume spike filter
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = max(50, 20)  # EMA50 and Donchian20
    
    for i in range(start_idx, n):
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend from 12h EMA(50): price above EMA = uptrend, below = downtrend
        uptrend = close[i] > ema_12h_aligned[i]
        downtrend = close[i] < ema_12h_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout above upper Donchian band with volume confirmation and uptrend
            if close[i] > donchian_upper[i] and volume_confirmation[i] and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Breakdown below lower Donchian band with volume confirmation and downtrend
            elif close[i] < donchian_lower[i] and volume_confirmation[i] and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below lower Donchian band (reversal) OR trend changes
            if close[i] < donchian_lower[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price breaks above upper Donchian band (reversal) OR trend changes
            if close[i] > donchian_upper[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals