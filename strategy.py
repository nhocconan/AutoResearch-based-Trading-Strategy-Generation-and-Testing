#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation
# Donchian channels provide robust trend-following structure, 1w EMA34 ensures alignment with weekly trend
# Volume confirmation filters false breakouts. Designed for 1d timeframe targeting 7-25 trades/year (30-100 total over 4 years)
# Uses discrete position sizing (0.25) to balance return and drawdown control
# Works in bull markets (breakout above upper Donchian + 1w EMA34 up) and bear markets (breakout below lower Donchian + 1w EMA34 down)

name = "1d_Donchian20_1wEMA34_Trend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # 1w data for trend filter (EMA34)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need enough for EMA calculation
        return np.zeros(n)
    
    # 1w EMA34 calculation
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Donchian channels (20-period) from 1d data
    # Upper band = highest high over past 20 days
    # Lower band = lowest low over past 20 days
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
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 1w EMA34
        uptrend = close[i] > ema_34_1w_aligned[i]
        downtrend = close[i] < ema_34_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout above upper Donchian with volume confirmation and uptrend
            if high[i] > donchian_upper[i] and volume_confirmation[i] and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Breakout below lower Donchian with volume confirmation and downtrend
            elif low[i] < donchian_lower[i] and volume_confirmation[i] and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below lower Donchian (reversal) OR trend changes
            if low[i] < donchian_lower[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price breaks above upper Donchian (reversal) OR trend changes
            if high[i] > donchian_upper[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals