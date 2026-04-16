#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
# Long when price breaks above Donchian upper band AND close > 12h EMA50 (uptrend) AND volume > 1.5x average.
# Short when price breaks below Donchian lower band AND close < 12h EMA50 (downtrend) AND volume > 1.5x average.
# Exit when price crosses the opposite Donchian band or 12h EMA50.
# Uses discrete position size 0.25. Donchian provides clear structure, EMA50 filters trend direction,
# volume confirmation ensures breakout validity. 4h timeframe targets 75-200 total trades over 4 years
# (19-50/year) to minimize fee drag while capturing sustained moves in both bull and bear markets.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data once before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # === 12h Indicators: EMA50 for trend filter ===
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    signals = np.zeros(n)
    
    # Warmup: Donchian(20) needs 20 bars, EMA50 needs 50
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if np.isnan(ema50_aligned[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol = volume[i]
        ema50 = ema50_aligned[i]
        
        # Calculate Donchian channels (20-period)
        lookback_start = max(0, i - 19)
        highest_high = np.max(high[lookback_start:i+1])
        lowest_low = np.min(low[lookback_start:i+1])
        
        # Volume filter: current volume > 1.5x 20-period average
        if i >= 20:
            vol_ma = np.mean(volume[i-19:i+1])
            volume_filter = vol > 1.5 * vol_ma
        else:
            volume_filter = False
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when price < Donchian lower band OR price < 12h EMA50 (trend break)
            if (price < lowest_low) or (price < ema50):
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when price > Donchian upper band OR price > 12h EMA50 (trend break)
            if (price > highest_high) or (price > ema50):
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: price > Donchian upper band AND price > 12h EMA50 (uptrend) AND volume confirmation
            if (price > highest_high) and (price > ema50) and volume_filter:
                signals[i] = 0.25
                position = 1
            
            # SHORT: price < Donchian lower band AND price < 12h EMA50 (downtrend) AND volume confirmation
            elif (price < lowest_low) and (price < ema50) and volume_filter:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "4h_Donchian20_VolumeConfirmation_12hEMA50_TrendFilter_V1"
timeframe = "4h"
leverage = 1.0