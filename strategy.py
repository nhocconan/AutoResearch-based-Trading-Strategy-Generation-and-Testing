#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and ATR-based trailing stop.
# Long when price > upper Donchian + volume > 1.5x 20-period median volume.
# Short when price < lower Donchian + same volume condition.
# Exit when price reverses 1.5x ATR from extreme (highest high for longs, lowest low for shorts).
# Uses discrete position size 0.25. Session filter: 08-20 UTC.
# Target: 75-200 total trades over 4 years (19-50/year). Uses 4h for all calculations.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute hour for session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    
    # Get 4h data once before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # === 4h Indicators ===
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    vol_4h = df_4h['volume'].values
    
    # Donchian channels (20-period)
    upper_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Volume median (20-period)
    vol_median_20 = pd.Series(vol_4h).rolling(window=20, min_periods=20).median().values
    
    # ATR (14-period) for trailing stop
    tr1 = pd.Series(high_4h).rolling(2).apply(lambda x: x[1] - x[0] if len(x)==2 else 0, raw=True).shift(1).fillna(0)
    tr2 = abs(pd.Series(high_4h).shift(1) - pd.Series(low_4h))
    tr3 = abs(pd.Series(low_4h).shift(1) - pd.Series(close_4h))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align all indicators to primary timeframe (4h)
    upper_20_aligned = align_htf_to_ltf(prices, df_4h, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_4h, lower_20)
    vol_median_aligned = align_htf_to_ltf(prices, df_4h, vol_median_20)
    vol_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_4h)
    atr_14_aligned = align_htf_to_ltf(prices, df_4h, atr_14)
    high_4h_aligned = align_htf_to_ltf(prices, df_4h, high_4h)
    low_4h_aligned = align_htf_to_ltf(prices, df_4h, low_4h)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(20, 14)  # Donchian(20), ATR(14)
    
    # Track position state and extremes for trailing stop
    position = 0  # 0: flat, 1: long, -1: short
    long_extreme = 0.0   # highest high since entering long
    short_extreme = 0.0  # lowest low since entering short
    
    for i in range(warmup, n):
        # Session filter: 08-20 UTC only
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            position = 0
            long_extreme = 0.0
            short_extreme = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(vol_median_aligned[i]) or np.isnan(vol_4h_aligned[i]) or 
            np.isnan(atr_14_aligned[i]) or np.isnan(high_4h_aligned[i]) or 
            np.isnan(low_4h_aligned[i])):
            signals[i] = 0.0
            position = 0
            long_extreme = 0.0
            short_extreme = 0.0
            continue
        
        # Current values (aligned)
        upper = upper_20_aligned[i]
        lower = lower_20_aligned[i]
        vol_median = vol_median_aligned[i]
        vol_4h = vol_4h_aligned[i]
        atr = atr_14_aligned[i]
        high_price = high_4h_aligned[i]
        low_price = low_4h_aligned[i]
        
        # Price levels
        price = close[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        if position == 1:  # long position
            # Update long extreme
            long_extreme = max(long_extreme, high_price)
            # Exit when price drops 1.5*ATR from extreme
            if price < long_extreme - 1.5 * atr:
                exit_signal = True
        elif position == -1:  # short position
            # Update short extreme
            short_extreme = min(short_extreme, low_price)
            # Exit when price rises 1.5*ATR from extreme
            if price > short_extreme + 1.5 * atr:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            long_extreme = 0.0
            short_extreme = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Volume spike filter: current 4h volume > 1.5x median volume
            volume_spike = vol_4h > (vol_median * 1.5)
            
            # LONG CONDITIONS
            # Price breaks above upper Donchian band AND volume spike
            if price > upper and volume_spike:
                signals[i] = 0.25
                position = 1
                long_extreme = high_price  # initialize extreme
            
            # SHORT CONDITIONS
            # Price breaks below lower Donchian band AND volume spike
            elif price < lower and volume_spike:
                signals[i] = -0.25
                position = -1
                short_extreme = low_price  # initialize extreme
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "4h_Donchian20_4hVolumeSpike1.5x_ATRTrail1.5x_v1"
timeframe = "4h"
leverage = 1.0