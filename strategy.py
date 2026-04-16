#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA(50) trend filter and 1d volume spike (2x median)
# Long when price > Donchian upper AND price > 12h EMA50 AND 1d volume > 2x 20-period median
# Short when price < Donchian lower AND price < 12h EMA50 AND 1d volume > 2x 20-period median
# Exit when price crosses 12h EMA50 (mean reversion to trend)
# Uses discrete position size 0.25 to limit fee drag. Target: 75-200 total trades over 4 years.
# Combines price channel breakout with trend and volume filters for robustness across bull/bear markets.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data once before loop for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Indicators: Volume median (20-period) ===
    volume_1d = df_1d['volume'].values
    vol_median_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).median().values
    vol_median_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_median_20_1d)
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # === 12h Indicators: EMA (50-period) for trend filter ===
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Get 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # === 4h Indicators: Donchian Channel (20-period) ===
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Donchian upper/lower
    donch_upper = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_lower = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to primary timeframe
    donch_upper_aligned = align_htf_to_ltf(prices, df_4h, donch_upper)
    donch_lower_aligned = align_htf_to_ltf(prices, df_4h, donch_lower)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(30, 20, 50, 20)  # 1d volume, 4h Donchian, 12h EMA
    
    # Track position state for exits
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            if position != 0:
                position = 0  # force flat outside session
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(donch_upper_aligned[i]) or np.isnan(donch_lower_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_median_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current 1d volume (aligned)
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        if np.isnan(vol_1d_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 1d volume > 2x 20-period 1d volume median
        vol_threshold = vol_median_20_1d_aligned[i] * 2.0
        vol_confirm = vol_1d_aligned[i] > vol_threshold
        
        # Price levels
        price = close[i]
        upper = donch_upper_aligned[i]
        lower = donch_lower_aligned[i]
        ema50 = ema_50_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        if position == 1:  # long position
            # Exit when price crosses below 12h EMA50 (trend change)
            if price < ema50:
                exit_signal = True
        elif position == -1:  # short position
            # Exit when price crosses above 12h EMA50 (trend change)
            if price > ema50:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG CONDITIONS
            # Price breaks above Donchian upper AND volume confirmation AND price > EMA50
            if price > upper and vol_confirm and price > ema50:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT CONDITIONS
            # Price breaks below Donchian lower AND volume confirmation AND price < EMA50
            elif price < lower and vol_confirm and price < ema50:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "4h_Donchian20_12hEMA50_1dVolume2x_v1"
timeframe = "4h"
leverage = 1.0