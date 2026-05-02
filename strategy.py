#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA(34) trend filter and volume confirmation
# Uses weekly EMA(34) for major trend direction to avoid counter-trend trades
# Donchian(20) breakout on 1d provides structure with proven edge on SOL/ETH/BTC
# Volume confirmation (1.5x 20-period average) ensures participation and reduces false breakouts
# Discrete position sizing 0.25 balances risk and minimizes fee churn
# Targets 8-20 trades/year (30-80 total over 4 years) to stay within fee drag limits
# Works in both bull and bear markets by aligning with higher timeframe weekly trend

name = "1d_Donchian20_1wEMA34_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    # Calculate 1w EMA(34) for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 1d Donchian(20) channels
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate 1d volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Donchian and volume MA)
    start_idx = 40  # 20 for Donchian + 20 for volume MA + buffer
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(high_ma[i]) or 
            np.isnan(low_ma[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from 1w EMA
        uptrend = close[i] > ema_34_1w_aligned[i]
        downtrend = close[i] < ema_34_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian high AND uptrend AND volume confirm
            if (close[i] > high_ma[i] and 
                uptrend and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND downtrend AND volume confirm
            elif (close[i] < low_ma[i] and 
                  downtrend and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below Donchian low OR trend reverses to downtrend
            if (close[i] < low_ma[i] or 
                not uptrend):  # exited if price closes below 1w EMA
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR trend reverses to uptrend
            if (close[i] > high_ma[i] or 
                not downtrend):  # exited if price closes above 1w EMA
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals