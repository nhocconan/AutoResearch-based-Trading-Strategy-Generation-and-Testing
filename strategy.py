#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 4h volume spike filter (volume > 1.8x 20-period median) and 12h EMA50 trend filter (price > EMA50)
# Long when price > Donchian upper band AND 4h volume > 1.8x 20-period median volume AND 12h close > 12h EMA50
# Short when price < Donchian lower band AND 4h volume > 1.8x 20-period median volume AND 12h close < 12h EMA50
# Exit when price crosses Donchian middle band
# Uses discrete position size 0.25. Target: 50-150 total trades over 4 years (12-37/year).
# Combines price channel breakout with volume confirmation and 12h trend filter for robustness in both bull and bear markets.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data once before loop for Donchian levels and EMA trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # === 12h Indicators: Donchian channels (20-period) and EMA50 trend ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Donchian channels
    donchian_upper_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_lower_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_middle_20 = (donchian_upper_20 + donchian_lower_20) / 2.0
    
    # EMA50 trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 4h data for volume confirmation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # === 4h Indicators: Volume median for confirmation ===
    volume_4h = df_4h['volume'].values
    volume_median_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).median().values
    
    # Align all indicators to primary timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper_20)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower_20)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_12h, donchian_middle_20)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    volume_median_aligned = align_htf_to_ltf(prices, df_4h, volume_median_20_4h)
    
    # Align 4h volume for volume confirmation
    volume_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_4h)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(30, 20, 50)  # 12h Donchian, 12h EMA50, 4h volume median
    
    # Track position state for exits
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(volume_median_aligned[i]) or np.isnan(volume_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current values (aligned)
        upper = donchian_upper_aligned[i]
        lower = donchian_lower_aligned[i]
        middle = donchian_middle_aligned[i]
        vol_median = volume_median_aligned[i]
        ema50_12h = ema_50_12h_aligned[i]
        vol_4h = volume_4h_aligned[i]
        
        # Price levels
        price = close[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        if position == 1:  # long position
            # Exit when price crosses below Donchian middle band (mean reversion)
            if price < middle:
                exit_signal = True
        elif position == -1:  # short position
            # Exit when price crosses above Donchian middle band (mean reversion)
            if price > middle:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Volume filter: current 4h volume > 1.8x 20-period median volume (balanced filter)
            vol_threshold = vol_median * 1.8
            vol_confirm = vol_4h > vol_threshold
            
            # LONG CONDITIONS
            # Price breaks above Donchian upper band AND volume confirmation AND 12h uptrend
            if price > upper and vol_confirm and close_12h_aligned[i] > ema50_12h:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT CONDITIONS
            # Price breaks below Donchian lower band AND volume confirmation AND 12h downtrend
            elif price < lower and vol_confirm and close_12h_aligned[i] < ema50_12h:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

# Pre-compute aligned 12h close for trend comparison
df_12h_close = None
def generate_signals(prices):
    global df_12h_close
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data once before loop for Donchian levels and EMA trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # === 12h Indicators: Donchian channels (20-period) and EMA50 trend ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Donchian channels
    donchian_upper_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_lower_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_middle_20 = (donchian_upper_20 + donchian_lower_20) / 2.0
    
    # EMA50 trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 4h data for volume confirmation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # === 4h Indicators: Volume median for confirmation ===
    volume_4h = df_4h['volume'].values
    volume_median_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).median().values
    
    # Align all indicators to primary timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper_20)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower_20)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_12h, donchian_middle_20)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    volume_median_aligned = align_htf_to_ltf(prices, df_4h, volume_median_20_4h)
    
    # Align 4h volume for volume confirmation
    volume_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_4h)
    # Align 12h close for trend comparison
    close_12h_aligned = align_htf_to_ltf(prices, df_12h, close_12h)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(30, 20, 50)  # 12h Donchian, 12h EMA50, 4h volume median
    
    # Track position state for exits
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(volume_median_aligned[i]) or np.isnan(volume_4h_aligned[i]) or
            np.isnan(close_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current values (aligned)
        upper = donchian_upper_aligned[i]
        lower = donchian_lower_aligned[i]
        middle = donchian_middle_aligned[i]
        vol_median = volume_median_aligned[i]
        ema50_12h = ema_50_12h_aligned[i]
        vol_4h = volume_4h_aligned[i]
        close_12h_val = close_12h_aligned[i]
        
        # Price levels
        price = close[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        if position == 1:  # long position
            # Exit when price crosses below Donchian middle band (mean reversion)
            if price < middle:
                exit_signal = True
        elif position == -1:  # short position
            # Exit when price crosses above Donchian middle band (mean reversion)
            if price > middle:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Volume filter: current 4h volume > 1.8x 20-period median volume (balanced filter)
            vol_threshold = vol_median * 1.8
            vol_confirm = vol_4h > vol_threshold
            
            # LONG CONDITIONS
            # Price breaks above Donchian upper band AND volume confirmation AND 12h uptrend
            if price > upper and vol_confirm and close_12h_val > ema50_12h:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT CONDITIONS
            # Price breaks below Donchian lower band AND volume confirmation AND 12h downtrend
            elif price < lower and vol_confirm and close_12h_val < ema50_12h:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "12h_Donchian20_4hVolumeSpike1.8x_12hEMA50_v1"
timeframe = "12h"
leverage = 1.0