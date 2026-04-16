#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using Donchian channel breakout (20) with 1d EMA50 trend filter and volume confirmation.
# Long when price breaks above upper Donchian(20) AND close > 1d EMA50 AND volume > 1.5x 20-period average volume.
# Short when price breaks below lower Donchian(20) AND close < 1d EMA50 AND volume > 1.5x 20-period average volume.
# Exit when price crosses the 1d EMA50 (trend reversal) or Donchian middle line.
# Uses discrete position size 0.25. Donchian provides clear breakout levels, EMA50 filters trend direction,
# volume confirmation reduces false breakouts. 12h timeframe targets 50-150 total trades over 4 years (12-37/year)
# to minimize fee drag. Works in bull markets (catch breakouts in uptrend) and bear markets (catch breakdowns in downtrend).

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data once before loop for Donchian channel
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 40:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Get 1d data once before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # === 12h Indicators: Donchian Channel (20) ===
    # Upper band = highest high over 20 periods
    # Lower band = lowest low over 20 periods
    # Middle band = (upper + lower) / 2
    high_series = pd.Series(high_12h)
    low_series = pd.Series(low_12h)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # === 1d Indicators: EMA50 for trend filter ===
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # === Volume confirmation: 20-period average volume ===
    volume_series = pd.Series(volume)
    avg_volume_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to primary timeframe (12h)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_12h, donchian_middle)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    avg_volume_aligned = align_htf_to_ltf(prices, df_12h, avg_volume_20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 60  # EMA50 and Donchian need sufficient warmup
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(ema50_aligned[i]) or 
            np.isnan(avg_volume_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        price = close[i]
        vol = volume[i]
        upper = donchian_upper_aligned[i]
        lower = donchian_lower_aligned[i]
        middle = donchian_middle_aligned[i]
        ema50 = ema50_aligned[i]
        avg_vol = avg_volume_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when price < EMA50 (trend break) OR price < middle (mean reversion)
            if (price < ema50) or (price < middle):
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when price > EMA50 (trend break) OR price > middle (mean reversion)
            if (price > ema50) or (price > middle):
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: price breaks above upper Donchian AND price > EMA50 (uptrend) AND volume confirmation
            if (price > upper) and (price > ema50) and volume_confirm:
                signals[i] = 0.25
                position = 1
            
            # SHORT: price breaks below lower Donchian AND price < EMA50 (downtrend) AND volume confirmation
            elif (price < lower) and (price < ema50) and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "12h_Donchian20_VolumeConfirmation_1dEMA50_TrendFilter_V1"
timeframe = "12h"
leverage = 1.0