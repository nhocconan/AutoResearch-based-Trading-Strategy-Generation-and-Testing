#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume spike confirmation.
# Long when price breaks above 20-period high AND close > 1w EMA50 AND volume > 2.0x 20-period average.
# Short when price breaks below 20-period low AND close < 1w EMA50 AND volume > 2.0x 20-period average.
# Exit when price crosses the 20-period midpoint (upper+lower)/2.
# Uses discrete position size 0.25. Designed to capture strong breakouts in trending markets.
# Target: 50-100 total trades over 4 years (12-25/year) to minimize fee drag while maintaining edge.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d Indicators: Donchian Channel (20-period) ===
    lookback = 20
    highest = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    upper = highest
    lower = lowest
    midpoint = (upper + lower) / 2
    
    # === 1w Indicators: EMA50 trend filter ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    trend_up = close > ema_50_1w_aligned
    trend_down = close < ema_50_1w_aligned
    
    # === Volume Spike: volume > 2.0x 20-period average ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for EMA50)
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(midpoint[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_spike[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price crosses below midpoint
            if price < midpoint[i]:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price crosses above midpoint
            if price > midpoint[i]:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above upper AND uptrend AND volume spike
            if price > upper[i] and trend_up[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below lower AND downtrend AND volume spike
            elif price < lower[i] and trend_down[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "1d_Donchian20_1wEMA50_VolumeSpike_V1"
timeframe = "1d"
leverage = 1.0