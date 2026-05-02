#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation
# Donchian(20) breakout captures strong momentum moves in both bull and bear markets
# 1w EMA34 ensures alignment with higher timeframe trend to avoid counter-trend trades
# Volume confirmation filters out false breakouts
# Designed for 1d timeframe targeting 7-25 trades/year (30-100 total over 4 years)
# Uses discrete position sizing (0.25) to minimize fee churn and control drawdown
# Works in bull markets (breakout above upper band + 1w trend up) and bear markets (breakout below lower band + 1w trend down)

name = "1d_Donchian20_1wEMA34_Trend_Volume"
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
    
    # 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # 1w EMA34 trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # 1d Donchian(20) - need 20 periods of 1d data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels on 1d data
    donchian_high = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1d timeframe (same as original since 1d->1d)
    donchian_high_aligned = donchian_high  # Already 1d aligned
    donchian_low_aligned = donchian_low    # Already 1d aligned
    
    # Volume confirmation on 1d timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmation = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 1w EMA34
        bullish_bias = close[i] > ema_34_1w_aligned[i]
        bearish_bias = close[i] < ema_34_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout above upper Donchian band with volume confirmation and 1w trend up
            if close[i] > donchian_high_aligned[i] and bullish_bias and volume_confirmation[i]:
                signals[i] = 0.25
                position = 1
            # Short: Breakout below lower Donchian band with volume confirmation and 1w trend down
            elif close[i] < donchian_low_aligned[i] and bearish_bias and volume_confirmation[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below lower Donchian band (reversal) OR 1w trend turns bearish
            if close[i] < donchian_low_aligned[i] or not bullish_bias:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price breaks above upper Donchian band (reversal) OR 1w trend turns bullish
            if close[i] > donchian_high_aligned[i] or not bearish_bias:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals