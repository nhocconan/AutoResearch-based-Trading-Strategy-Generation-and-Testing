#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d volume spike and 1d chop regime filter.
# Long when Williams %R(14) < -80 (oversold) AND volume > 1.3x 20-period 1d average AND chop > 61.8 (ranging market).
# Short when Williams %R(14) > -20 (overbought) AND volume > 1.3x 20-period 1d average AND chop > 61.8.
# Exit when Williams %R crosses above -50 (for longs) or below -50 (for shorts).
# Uses discrete position size 0.25. Designed to capture mean reversion in ranging markets with volume confirmation.
# Target: 80-180 total trades over 4 years (20-45/year) to balance opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Indicators: Williams %R (14-period) ===
    highest_high_6h = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_6h = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_6h - close) / (highest_high_6h - lowest_low_6h)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high_6h - lowest_low_6h) == 0, -50, williams_r)
    
    # === 1d Indicators: Volume Spike (volume > 1.3x 20-period average) ===
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (1.3 * vol_ma_1d_aligned)
    
    # === 1d Indicators: Choppiness Index (CHOP) > 61.8 (ranging market) ===
    atr_1d_list = []
    for i in range(len(df_1d)):
        if i < 14:
            atr_1d_list.append(np.nan)
        else:
            tr = np.maximum(
                df_1d['high'].iloc[i] - df_1d['low'].iloc[i],
                np.maximum(
                    np.abs(df_1d['high'].iloc[i] - df_1d['close'].iloc[i-1]),
                    np.abs(df_1d['low'].iloc[i] - df_1d['close'].iloc[i-1])
                )
            )
            if i == 14:
                atr_1d_list.append(tr)
            else:
                atr_1d_list.append((atr_1d_list[-1] * 13 + tr) / 14)
    atr_1d = np.array(atr_1d_list)
    
    highest_high_14d = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low_14d = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_1d * np.sqrt(14) / (highest_high_14d - lowest_low_14d)) / np.log10(14)
    # Handle division by zero and invalid values
    chop = np.where((highest_high_14d - lowest_low_14d) == 0, 50, chop)
    chop = np.where(np.isnan(chop), 50, chop)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    chop_filter = chop_aligned > 61.8
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(williams_r[i]) or np.isnan(volume_spike[i]) or np.isnan(chop_filter[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        wr = williams_r[i]
        vol_spike = volume_spike[i]
        is_chop = chop_filter[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if Williams %R crosses above -50
            if wr > -50:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if Williams %R crosses below -50
            if wr < -50:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Williams %R oversold (< -80) AND volume spike AND choppy market
            if wr < -80 and vol_spike and is_chop:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Williams %R overbought (> -20) AND volume spike AND choppy market
            elif wr > -20 and vol_spike and is_chop:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_WilliamsR_MeanReversion_1dVolumeSpike_1dChop_V1"
timeframe = "6h"
leverage = 1.0