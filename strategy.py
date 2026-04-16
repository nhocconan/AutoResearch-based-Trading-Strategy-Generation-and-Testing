#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R (14) mean reversion with 1w EMA200 trend filter and 1d volume spike confirmation
# Long when Williams %R < -80 (oversold) AND price > weekly EMA200 AND 1d volume > 1.5x 20-period median volume
# Short when Williams %R > -20 (overbought) AND price < weekly EMA200 AND 1d volume > 1.5x 20-period median volume
# Exit when Williams %R crosses back above -50 (for longs) or below -50 (for shorts)
# Uses discrete position size 0.25. Target: 50-150 total trades over 4 years (12-37/year).
# Williams %R is effective at identifying exhaustion points in both trending and ranging markets.
# Weekly EMA200 ensures we trade with the higher timeframe trend, avoiding counter-trend moves.
# Volume confirmation filters out low-conviction breakouts/reversals.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop for Williams %R and volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Indicators: Williams %R (14) and volume median ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14) * -100
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # Volume median for scaling
    volume_median_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).median().values
    
    # Get weekly data for trend filter (EMA200)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # === Weekly Indicators: EMA200 trend filter ===
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align all indicators to primary timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    volume_median_aligned = align_htf_to_ltf(prices, df_1d, volume_median_20_1d)
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Align 1d volume for volume confirmation
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(30, 14, 20, 200)  # 1d Williams %R, 1d volume median, weekly EMA200
    
    # Track position state for exits
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(volume_median_aligned[i]) or 
            np.isnan(ema_200_1w_aligned[i]) or np.isnan(volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current values (aligned)
        wr = williams_r_aligned[i]
        vol_median = volume_median_aligned[i]
        weekly_ema200 = ema_200_1w_aligned[i]
        vol_1d = volume_1d_aligned[i]
        
        # Price levels
        price = close[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        if position == 1:  # long position
            # Exit when Williams %R crosses back above -50 (recovering from oversold)
            if wr > -50:
                exit_signal = True
        elif position == -1:  # short position
            # Exit when Williams %R crosses back below -50 (declining from overbought)
            if wr < -50:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Volume filter: current 1d volume > 1.5x 20d median volume
            vol_threshold = vol_median * 1.5
            vol_confirm = vol_1d > vol_threshold
            
            # LONG CONDITIONS
            # Williams %R < -80 (oversold) AND price > weekly EMA200 AND volume confirmation
            if wr < -80 and price > weekly_ema200 and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT CONDITIONS
            # Williams %R > -20 (overbought) AND price < weekly EMA200 AND volume confirmation
            elif wr > -20 and price < weekly_ema200 and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "6h_WilliamsR14_1wEMA200_1dVolumeSpike1.5x_v1"
timeframe = "6h"
leverage = 1.0