#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Donchian breakout captures strong momentum moves, effective in both bull and bear markets
# 1w EMA50 ensures alignment with higher timeframe trend to avoid range-bound whipsaws
# Volume confirmation filters weak breakouts, institutional participation signal
# Designed for 1d timeframe targeting 7-25 trades/year (30-100 total over 4 years)
# Uses discrete position sizing (0.30) to minimize fee churn and control drawdown
# Works in bull markets (breakout above upper band + 1w EMA50 up-trend) and bear markets (breakout below lower band + 1w EMA50 down-trend)

name = "1d_Donchian20_1wEMA50_Trend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need enough for EMA calculation
        return np.zeros(n)
    
    # 1w EMA50 calculation
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 1d Donchian(20) channels
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min()
    upper_band = high_roll.values
    lower_band = low_roll.values
    
    # Volume confirmation
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmation = volume > (1.5 * vol_ema_20)  # Volume spike filter
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i]) or np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 1w EMA50 (trending market filter)
        # Uptrend: price above EMA50, Downtrend: price below EMA50
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout above upper band with volume confirmation and uptrend
            if close[i] > upper_band[i] and uptrend and volume_confirmation[i]:
                signals[i] = 0.30
                position = 1
            # Short: Breakout below lower band with volume confirmation and downtrend
            elif close[i] < lower_band[i] and downtrend and volume_confirmation[i]:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below lower band (reversal) OR trend changes to downtrend
            if close[i] < lower_band[i] or downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit: Price breaks above upper band (reversal) OR trend changes to uptrend
            if close[i] > upper_band[i] or uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals