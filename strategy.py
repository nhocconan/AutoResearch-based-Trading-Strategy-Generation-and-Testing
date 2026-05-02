#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w trend filter and ATR volatility filter
# Uses 1d Donchian channels for breakout signals, 1w EMA(50) for trend direction
# ATR(14) > 1.5x ATR(50) ensures sufficient volatility to avoid choppy markets
# Only takes breakouts in the direction of the 1w trend to avoid counter-trend whipsaws
# Discrete position sizing 0.25 balances risk and minimizes fee churn
# Targets 30-100 total trades over 4 years (7-25/year) to stay within fee drag limits
# Works in both bull and bear markets by aligning with higher timeframe trend
# 1w trend filter provides strong directional bias suitable for 1d timeframe

name = "1d_Donchian20_1wTrend_ATRVol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Calculate 1w EMA(50) for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1d Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate ATR(14) and ATR(50) for volatility filter
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    high_close[0] = 0
    low_close[0] = 0
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Volatility filter: ATR(14) > 1.5 * ATR(50)
    vol_filter = atr_14 > (atr_50 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Donchian, ATR and EMA)
    start_idx = 60  # max(20 for Donchian, 50 for ATR/EMA) + buffer
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from 1w EMA
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian upper band AND uptrend AND vol filter
            if (close[i] > high_roll[i] and 
                uptrend and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower band AND downtrend AND vol filter
            elif (close[i] < low_roll[i] and 
                  downtrend and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price breaks below Donchian lower band OR trend reverses to downtrend
            if (close[i] < low_roll[i] or 
                not uptrend):  # exited if price closes below 1w EMA
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian upper band OR trend reverses to uptrend
            if (close[i] > high_roll[i] or 
                not downtrend):  # exited if price closes above 1w EMA
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals