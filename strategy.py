#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
# Long when price breaks above Donchian upper channel (20-period high) AND 12h EMA50 is rising AND current volume > 1.5x 20-period median volume.
# Short when price breaks below Donchian lower channel (20-period low) AND 12h EMA50 is falling AND same volume condition.
# Exit via ATR(10) trailing stop: long exits when price < highest high since entry - 2.5*ATR, short exits when price > lowest low since entry + 2.5*ATR.
# Uses discrete position size 0.25. Target: 75-200 total trades over 4 years (19-50/year).
# Donchian captures structure, EMA50 filters trend direction, volume confirms conviction, ATR stop manages risk.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data once before loop for Donchian channels and ATR
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Get 12h data once before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # === 4h Indicators: Donchian(20), ATR(10) ===
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # ATR(10)
    tr1 = pd.Series(high_4h - low_4h)
    tr2 = pd.Series(np.abs(high_4h - np.roll(close_4h, 1)))
    tr3 = pd.Series(np.abs(low_4h - np.roll(close_4h, 1)))
    tr2.iloc[0] = tr1.iloc[0]
    tr3.iloc[0] = tr1.iloc[0]
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_10 = tr.rolling(window=10, min_periods=10).mean().values
    
    # Donchian(20)
    donchian_upper = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # === 12h Indicators: EMA50 trend filter ===
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    # EMA slope: rising if current > previous, falling if current < previous
    ema_slope = np.diff(ema_50, prepend=ema_50[0])
    
    # Get volume median for spike filter
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # Align all indicators to primary timeframe (4h)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    atr_aligned = align_htf_to_ltf(prices, df_4h, atr_10)
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    ema_slope_aligned = align_htf_to_ltf(prices, df_12h, ema_slope)
    vol_median_aligned = align_htf_to_ltf(prices, prices, vol_median_20)  # align to self
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(20, 10, 50, 20)  # Donchian(20), ATR(10), EMA(50), volume median(20)
    
    # Track position state for trailing stops
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(atr_aligned[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(ema_slope_aligned[i]) or 
            np.isnan(vol_median_aligned[i])):
            signals[i] = 0.0
            position = 0
            highest_since_entry = 0.0
            lowest_since_entry = 0.0
            continue
        
        # Current values (aligned)
        dc_upper = donchian_upper_aligned[i]
        dc_lower = donchian_lower_aligned[i]
        atr = atr_aligned[i]
        ema50 = ema_50_aligned[i]
        emaslope = ema_slope_aligned[i]
        vol_median = vol_median_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # Volume spike filter: current volume > 1.5x median volume
        volume_spike = vol > (vol_median * 1.5)
        
        # Breakout conditions
        breakout_up = price > dc_upper
        breakout_down = price < dc_lower
        
        # EMA trend filter: rising for long, falling for short
        ema_rising = emaslope > 0
        ema_falling = emaslope < 0
        
        # === EXIT LOGIC (trailing stop) ===
        exit_signal = False
        if position == 1:  # long position
            # Update highest high since entry
            if price > highest_since_entry:
                highest_since_entry = price
            # Exit when price drops below highest high - 2.5*ATR
            if price < highest_since_entry - 2.5 * atr:
                exit_signal = True
        elif position == -1:  # short position
            # Update lowest low since entry
            if price < lowest_since_entry:
                lowest_since_entry = price
            # Exit when price rises above lowest low + 2.5*ATR
            if price > lowest_since_entry + 2.5 * atr:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            highest_since_entry = 0.0
            lowest_since_entry = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG CONDITIONS
            # Breakout above upper channel, EMA rising, volume spike
            if breakout_up and ema_rising and volume_spike:
                signals[i] = 0.25
                position = 1
                highest_since_entry = price  # initialize trailing stop
            
            # SHORT CONDITIONS
            # Breakout below lower channel, EMA falling, volume spike
            elif breakout_down and ema_falling and volume_spike:
                signals[i] = -0.25
                position = -1
                lowest_since_entry = price  # initialize trailing stop
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "4h_Donchian20_EMA50_VolumeSpike1.5x_ATRTrail2.5_v1"
timeframe = "4h"
leverage = 1.0