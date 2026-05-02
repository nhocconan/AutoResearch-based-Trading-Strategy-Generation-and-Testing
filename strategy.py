#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Donchian channel breakouts capture strong momentum moves
# 1w EMA50 ensures alignment with weekly trend to avoid counter-trend whipsaws
# Volume confirmation (2x 20-period EMA) filters weak breakouts
# Designed for 1d timeframe targeting 7-25 trades/year (30-100 total over 4 years)
# Uses discrete position sizing (0.25) to minimize fee churn and control drawdown
# Works in bull markets (breakout above upper band + 1w EMA up-trend) and bear markets (breakout below lower band + 1w EMA down-trend)

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
    if len(df_1w) < 50:  # Need enough for EMA50 calculation
        return np.zeros(n)
    
    # 1w EMA50 calculation
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 1d Donchian channel (20-period)
    # Using rolling window on 1d data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Donchian upper and lower bands (20-period high/low)
    donchian_upper = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1d timeframe (same as original since we're on 1d)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    
    # Volume confirmation (2x 20-period EMA on 1d volume)
    vol_ema_20_1d = pd.Series(df_1d['volume']).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ema_20_1d)
    volume_confirmation = volume > (2.0 * vol_ema_20_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend from 1w EMA50 (price above/below EMA)
        weekly_uptrend = close[i] > ema_50_1w_aligned[i]
        weekly_downtrend = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout above upper Donchian band with volume confirmation and weekly uptrend
            if close[i] > donchian_upper_aligned[i] and weekly_uptrend and volume_confirmation[i]:
                signals[i] = 0.25
                position = 1
            # Short: Breakout below lower Donchian band with volume confirmation and weekly downtrend
            elif close[i] < donchian_lower_aligned[i] and weekly_downtrend and volume_confirmation[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below lower Donchian band (reversal) OR weekly trend turns down
            if close[i] < donchian_lower_aligned[i] or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price breaks above upper Donchian band (reversal) OR weekly trend turns up
            if close[i] > donchian_upper_aligned[i] or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals