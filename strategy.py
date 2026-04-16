#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and 12h EMA34 trend filter.
# Long when price > upper band, 1d volume > 1.5x its 20-period median, and 12h close > 12h EMA34.
# Short when price < lower band, same volume condition, and 12h close < 12h EMA34.
# Exit when price crosses middle band (mean reversion).
# Uses discrete position size 0.25. Target: 100-200 total trades over 4 years (25-50/year).
# Combines price channel breakout with volume confirmation and trend filter for robustness.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data once before loop for Donchian levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # === 4h Indicators: Donchian channels (20-period) ===
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Donchian channels
    donchian_upper_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_lower_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_middle_20 = (donchian_upper_20 + donchian_lower_20) / 2.0
    
    # Get daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === Daily Indicators: Volume confirmation ===
    vol_1d = df_1d['volume'].values
    vol_median_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).median().values
    
    # Get 12h data for trend filter (EMA34)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # === 12h Indicators: EMA34 trend filter ===
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all indicators to primary timeframe (4h)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper_20)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower_20)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_4h, donchian_middle_20)
    vol_median_aligned = align_htf_to_ltf(prices, df_1d, vol_median_20)
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Align daily volume for volume confirmation
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(30, 20, 30, 34)  # 4h Donchian, daily volume median, 12h EMA34
    
    # Track position state for exits
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(vol_median_aligned[i]) or 
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(vol_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current values (aligned)
        upper = donchian_upper_aligned[i]
        lower = donchian_lower_aligned[i]
        middle = donchian_middle_aligned[i]
        vol_median = vol_median_aligned[i]
        ema34_12h = ema_34_12h_aligned[i]
        daily_volume = vol_1d_aligned[i]
        
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
            # Volume confirmation: current daily volume > 1.5x its 20-period median
            volume_confirmed = daily_volume > (vol_median * 1.5)
            
            # LONG CONDITIONS
            # Price breaks above Donchian upper band AND volume confirmed AND 12h uptrend
            if price > upper and volume_confirmed and close_4h > ema34_12h:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT CONDITIONS
            # Price breaks below Donchian lower band AND volume confirmed AND 12h downtrend
            elif price < lower and volume_confirmed and close_4h < ema34_12h:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "4h_Donchian20_1dVolumeConf1.5x_12hEMA34_v1"
timeframe = "4h"
leverage = 1.0