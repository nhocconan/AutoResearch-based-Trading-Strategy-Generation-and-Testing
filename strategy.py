#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h EMA50 trend filter + volume confirmation
# Donchian breakout provides clear entry/exit levels with proven edge in trending markets
# 12h EMA50 ensures we only trade in the direction of the higher timeframe trend
# Volume confirmation (1.5x 20-period average) filters weak breakouts
# Discrete position sizing 0.25 balances risk and minimizes fee churn
# Targets 20-50 trades/year (80-200 total over 4 years) to stay within fee drag limits
# Works in both bull and bear markets by using trend filter and volatility-based stops

name = "4h_Donchian20_12hEMA50_Trend_VolumeConfirm_v1"
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
    
    # Load 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate Donchian channels (20-period) on 4h data
    # Upper channel: highest high over last 20 periods
    # Lower channel: lowest low over last 20 periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Donchian, EMA and volume MA)
    start_idx = 50  # max(20 for Donchian, 50 for EMA, 20 for volume) + buffer
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend from 12h EMA50
        uptrend = close[i] > ema50_12h_aligned[i]
        downtrend = close[i] < ema50_12h_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above Donchian upper channel AND uptrend AND volume confirmation
            if (close[i] > donchian_upper[i] and 
                uptrend and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower channel AND downtrend AND volume confirmation
            elif (close[i] < donchian_lower[i] and 
                  downtrend and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below Donchian lower channel (stop loss) OR 
            #        Price reaches middle of channel (partial profit)
            donchian_middle = (donchian_upper[i] + donchian_lower[i]) / 2
            if close[i] < donchian_lower[i]:
                signals[i] = 0.0
                position = 0
            elif close[i] >= donchian_middle:
                signals[i] = 0.125  # Half position (take profit)
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price breaks above Donchian upper channel (stop loss) OR
            #        Price reaches middle of channel (partial profit)
            donchian_middle = (donchian_upper[i] + donchian_lower[i]) / 2
            if close[i] > donchian_upper[i]:
                signals[i] = 0.0
                position = 0
            elif close[i] <= donchian_middle:
                signals[i] = -0.125  # Half position (take profit)
            else:
                signals[i] = -0.25
    
    return signals