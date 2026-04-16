#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
# Long when price breaks above Donchian upper band AND volume > 1.5x 20-period 1d average AND 1w EMA50 > EMA50 previous week (uptrend).
# Short when price breaks below Donchian lower band AND volume > 1.5x 20-period 1d average AND 1w EMA50 < EMA50 previous week (downtrend).
# Exit when price crosses the Donchian midpoint (upper+lower)/2.
# Uses discrete position size 0.25. Designed to capture breakouts in trending markets (both bull and bear).
# Target: 50-100 total trades over 4 years (12-25/year) to minimize fee drag while maintaining edge.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d Indicators: Donchian(20) channels (from previous bar) ===
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    # Donchian upper/lower band (20-period lookback)
    lookback = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(lookback, n):
        upper[i] = np.max(prev_high[i-lookback:i])
        lower[i] = np.min(prev_low[i-lookback:i])
    
    midpoint = (upper + lower) / 2  # Exit level
    
    # === 1d Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # === 1w Indicators: EMA50 trend filter (rising/falling) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # EMA50 calculation
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_prev = np.roll(ema_50, 1)
    ema_50_prev[0] = np.nan
    
    # EMA50 trend: rising if current > previous, falling if current < previous
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    ema_50_prev_aligned = align_htf_to_ltf(prices, df_1w, ema_50_prev)
    ema_rising = ema_50_aligned > ema_50_prev_aligned
    ema_falling = ema_50_aligned < ema_50_prev_aligned
    
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
            np.isnan(volume_spike[i]) or np.isnan(ema_rising[i]) or np.isnan(ema_falling[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        is_ema_rising = ema_rising[i]
        is_ema_falling = ema_falling[i]
        
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
            # LONG: Price breaks above upper band AND volume spike AND EMA50 rising
            if price > upper[i] and vol_spike and is_ema_rising:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price breaks below lower band AND volume spike AND EMA50 falling
            elif price < lower[i] and vol_spike and is_ema_falling:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "1d_Donchian20_1wEMA50_VolumeSpike_V1"
timeframe = "1d"
leverage = 1.0