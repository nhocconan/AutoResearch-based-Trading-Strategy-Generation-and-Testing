#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w volume confirmation and 1w EMA200 trend filter.
# Long when price > upper band, 1w volume > 1.5x median volume, and weekly close > weekly EMA200.
# Short when price < lower band, same volume condition, and weekly close < weekly EMA200.
# Exit when price crosses the middle band.
# Uses discrete position size 0.25. Target: 30-100 total trades over 4 years (7-25/year).
# Combines price channel breakout with volume spike filter and weekly trend filter for robustness in both bull and bear markets.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for Donchian levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Indicators: Donchian channel (20-period) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels
    upper_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    middle_20 = (upper_20 + lower_20) / 2.0
    
    # Get weekly data for volume confirmation and trend filter (EMA200)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # === Weekly Indicators: EMA200 trend filter ===
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # === Weekly Indicators: Volume median for confirmation ===
    volume_1w = df_1w['volume'].values
    vol_median_50 = pd.Series(volume_1w).rolling(window=50, min_periods=50).median().values
    
    # Align all indicators to primary timeframe (1d)
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    middle_20_aligned = align_htf_to_ltf(prices, df_1d, middle_20)
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    vol_median_50_aligned = align_htf_to_ltf(prices, df_1w, vol_median_50)
    
    # Align daily volume for volume spike detection
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, volume)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(20, 200, 50)  # Donchian(20), weekly EMA200, weekly volume median(50)
    
    # Track position state for exits
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(middle_20_aligned[i]) or np.isnan(ema_200_1w_aligned[i]) or 
            np.isnan(vol_median_50_aligned[i]) or np.isnan(vol_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current values (aligned)
        upper = upper_20_aligned[i]
        lower = lower_20_aligned[i]
        middle = middle_20_aligned[i]
        weekly_ema200 = ema_200_1w_aligned[i]
        weekly_vol_median = vol_median_50_aligned[i]
        daily_volume = vol_1d_aligned[i]
        
        # Get aligned daily close for proper trend comparison
        daily_close_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
        weekly_trend_up = daily_close_aligned[i] > weekly_ema200  # Using daily close vs weekly EMA for trend
        weekly_trend_down = daily_close_aligned[i] < weekly_ema200
        
        # Price levels
        price = close[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        if position == 1:  # long position
            # Exit when price crosses below middle band (mean reversion)
            if price < middle:
                exit_signal = True
        elif position == -1:  # short position
            # Exit when price crosses above middle band (mean reversion)
            if price > middle:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Volume spike filter: current daily volume > 1.5x weekly median volume
            # This ensures we only trade on high conviction moves
            volume_spike = daily_volume > (weekly_vol_median * 1.5)
            
            # LONG CONDITIONS
            # Price breaks above upper Donchian band AND volume spike AND weekly uptrend
            if price > upper and volume_spike and weekly_trend_up:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT CONDITIONS
            # Price breaks below lower Donchian band AND volume spike AND weekly downtrend
            elif price < lower and volume_spike and weekly_trend_down:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "1d_Donchian20_1wVolumeSpike_1wEMA200_v1"
timeframe = "1d"
leverage = 1.0