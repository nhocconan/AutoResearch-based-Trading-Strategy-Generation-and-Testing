#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Uses 1w HTF for EMA50 to capture weekly trend and reduce false breakouts in bear markets.
# Donchian breakout provides proven structure for capturing strong moves.
# Volume confirmation at 1.8x average ensures strong participation while limiting trades.
# Session filter (08-20 UTC) reduces noise trades during low-liquidity periods.
# Discrete sizing 0.25 to minimize fee churn. Works in bull/bear: trend filter ensures trades only with momentum.
# Target: 75-200 total trades over 4 years (19-50/year) to balance opportunity and fee drag.

name = "4h_Donchian20_Breakout_1wEMA50_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate Donchian(20) upper and lower bands from prior completed 4h bar
    # For 4h timeframe, we need to use the prior completed 4h bar's OHLC
    if len(prices) < 21:
        return np.zeros(n)
    
    # Get prior completed 4h bar's high/low (shift by 1 for 4h timeframe)
    prev_high = prices['high'].shift(1).values
    prev_low = prices['low'].shift(1).values
    
    # Donchian(20) upper and lower bands (proven breakout levels)
    donchian_upper = pd.Series(prev_high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(prev_low).rolling(window=20, min_periods=20).min().values
    
    # 1w EMA50 for trend filter (weekly trend)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: 1.8x 20-period average (strict threshold to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above Donchian upper AND price > 1w EMA50 AND volume spike
            if (close[i] > donchian_upper[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower AND price < 1w EMA50 AND volume spike
            elif (close[i] < donchian_lower[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price drops below Donchian lower OR price < 1w EMA50
            if close[i] < donchian_lower[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price rises above Donchian upper OR price > 1w EMA50
            if close[i] > donchian_upper[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals