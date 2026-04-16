#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using daily Donchian channel breakout with 1w EMA trend filter and volume confirmation.
# Long when price breaks above daily Donchian upper (20) with 1w EMA50 > EMA200 and volume > 1.5x 20-period average.
# Short when price breaks below daily Donchian lower (20) with 1w EMA50 < EMA200 and volume > 1.5x 20-period average.
# Exit when price returns to daily Donchian midpoint (mean reversion) or opposite Donchian level.
# Uses discrete position size 0.25. Daily Donchian provides structure from higher timeframe, 12h provides entry timing.
# Target: 50-150 total trades over 4 years (12-37/year) to balance edge and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once before loop for Donchian levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === Daily Indicators: Donchian Channel (20) based on prior day ===
    # Calculate using prior day's high, low (shift by 1 to use completed day only)
    phigh = np.roll(high_1d, 1)
    plow = np.roll(low_1d, 1)
    phigh[0] = np.nan
    plow[0] = np.nan
    
    # Donchian levels (based on prior day)
    donchian_upper = pd.Series(phigh).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(plow).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # Align daily Donchian levels to 12h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid)
    
    # Get weekly data once before loop for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # === Weekly Indicators: EMA50 and EMA200 for trend filter ===
    ema50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200 = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align weekly EMA to 12h timeframe
    ema50_aligned = align_htf_to_ltf(prices, df_1w, ema50)
    ema200_aligned = align_htf_to_ltf(prices, df_1w, ema200)
    
    # Volume moving average (20-period) on 12h
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(ema50_aligned[i]) or 
            np.isnan(ema200_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # Current values
        du = donchian_upper_aligned[i]
        dl = donchian_lower_aligned[i]
        dm = donchian_mid_aligned[i]
        ema50_val = ema50_aligned[i]
        ema200_val = ema200_aligned[i]
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price returns to daily Donchian midpoint or drops to Donchian lower
            if price <= dm or price <= dl:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price returns to daily Donchian midpoint or rises to Donchian upper
            if price >= dm or price >= du:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Trend filter: only trade when EMA50 > EMA200 for long, EMA50 < EMA200 for short
            trend_long = ema50_val > ema200_val
            trend_short = ema50_val < ema200_val
            
            # Volume filter: volume > 1.5x 20-period average
            vol_filter = vol > 1.5 * vol_ma
            
            # LONG: Price breaks above daily Donchian upper with trend and volume confirmation
            if (price > du) and trend_long and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below daily Donchian lower with trend and volume confirmation
            elif (price < dl) and trend_short and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "12h_1dDonchian20_1wEMA_VolumeConfirmation_V1"
timeframe = "12h"
leverage = 1.0