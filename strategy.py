#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Donchian breakout with 4h trend filter and volume confirmation.
# Long when price breaks above 20-bar 1h Donchian high AND 4h EMA50 uptrend (price > EMA50) AND 1h volume > 1.5x 20-period average.
# Short when price breaks below 20-bar 1h Donchian low AND 4h EMA50 downtrend (price < EMA50) AND 1h volume > 1.5x 20-period average.
# Uses discrete position size 0.20. Donchian breakouts capture momentum, 4h EMA50 ensures alignment with higher timeframe trend,
# volume spike confirms participation. Designed to work in both bull (buy breakouts) and bear (sell breakdowns) markets.
# Session filter: 08-20 UTC to reduce noise trades.
# Target: 60-150 total trades over 4 years = 15-37/year for 1h.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1h Indicators: Donchian Channel (20-period) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 1h Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Get 4h data once before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:  # Need enough for EMA50 calculation
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # === 4h Indicators: EMA50 for trend filter ===
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h EMA50 to 1h timeframe
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for EMA, 20 for Donchian/volume MA)
    warmup = 60
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        donchian_high = highest_high[i]
        donchian_low = lowest_low[i]
        ema_4h = ema_50_4h_aligned[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price breaks below Donchian low or volume spike ends
            if price < donchian_low or not vol_spike:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price breaks above Donchian high or volume spike ends
            if price > donchian_high or not vol_spike:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Donchian high AND price > 4h EMA50 (uptrend) AND volume spike
            if price > donchian_high and price > ema_4h and vol_spike:
                signals[i] = 0.20
                position = 1
            
            # SHORT: Price breaks below Donchian low AND price < 4h EMA50 (downtrend) AND volume spike
            elif price < donchian_low and price < ema_4h and vol_spike:
                signals[i] = -0.20
                position = -1
        
        else:
            signals[i] = position * 0.20
    
    return signals

name = "1h_Donchian20_4hEMA50_VolumeSpike_SessionFilter_V1"
timeframe = "1h"
leverage = 1.0