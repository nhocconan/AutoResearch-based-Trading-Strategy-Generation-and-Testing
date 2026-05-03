#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Extreme + 12h Supertrend + Volume Spike
# Long when Williams %R(14) < -80 (oversold) AND 12h Supertrend = bullish AND volume > 1.5x 20-period MA
# Short when Williams %R(14) > -20 (overbought) AND 12h Supertrend = bearish AND volume > 1.5x 20-period MA
# Williams %R identifies exhaustion points in both bull and bear markets.
# 12h Supertrend provides higher timeframe trend filter to avoid counter-trend trades.
# Volume spike confirms institutional participation at turning points.
# Designed for 6h timeframe to achieve 50-150 total trades over 4 years with proper risk control.

name = "6h_WilliamsR_Extreme_12hSupertrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_prices = prices['open'].values
    
    # Get 12h data for Supertrend calculation
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate Williams %R on primary timeframe (6h)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    period_wr = 14
    highest_high = pd.Series(high).rolling(window=period_wr, min_periods=period_wr).max().values
    lowest_low = pd.Series(low).rolling(window=period_wr, min_periods=period_wr).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate Supertrend on 12h timeframe
    # Supertrend uses ATR and multiplier
    period_atr = 10
    multiplier = 3.0
    
    # True Range
    tr1 = pd.Series(df_12h['high']).diff().abs()
    tr2 = (pd.Series(df_12h['high']) - pd.Series(df_12h['low'].shift())).abs()
    tr3 = (pd.Series(df_12h['low']) - pd.Series(df_12h['close'].shift())).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=period_atr, adjust=False, min_periods=period_atr).mean().values
    
    # Basic Upper and Lower Bands
    hl2 = (pd.Series(df_12h['high']) + pd.Series(df_12h['low'])) / 2
    upper_basic = hl2 + (multiplier * atr)
    lower_basic = hl2 - (multiplier * atr)
    
    # Final Upper and Lower Bands
    upper_band = np.zeros(len(df_12h))
    lower_band = np.zeros(len(df_12h))
    supertrend = np.zeros(len(df_12h))
    direction = np.ones(len(df_12h))  # 1 for uptrend, -1 for downtrend
    
    # Initialize
    upper_band[0] = upper_basic.iloc[0]
    lower_band[0] = lower_basic.iloc[0]
    supertrend[0] = upper_band[0]
    direction[0] = 1
    
    for i in range(1, len(df_12h)):
        # Upper Band
        if upper_basic.iloc[i] < upper_band[i-1] or df_12h['close'].iloc[i-1] > upper_band[i-1]:
            upper_band[i] = upper_basic.iloc[i]
        else:
            upper_band[i] = upper_band[i-1]
            
        # Lower Band
        if lower_basic.iloc[i] > lower_band[i-1] or df_12h['close'].iloc[i-1] < lower_band[i-1]:
            lower_band[i] = lower_basic.iloc[i]
        else:
            lower_band[i] = lower_band[i-1]
            
        # Supertrend and Direction
        if supertrend[i-1] == upper_band[i-1]:
            if df_12h['close'].iloc[i] <= upper_band[i]:
                supertrend[i] = upper_band[i]
                direction[i] = -1
            else:
                supertrend[i] = lower_band[i]
                direction[i] = 1
        else:
            if df_12h['close'].iloc[i] >= lower_band[i]:
                supertrend[i] = lower_band[i]
                direction[i] = 1
            else:
                supertrend[i] = upper_band[i]
                direction[i] = -1
    
    # Align 12h Supertrend direction to 6h timeframe
    supertrend_direction_aligned = align_htf_to_ltf(prices, df_12h, direction)
    
    # Volume spike detection (20-period volume MA on primary timeframe)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)  # Volume at least 1.5x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(williams_r[i]) or np.isnan(supertrend_direction_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        open_val = open_prices[i]
        vol_spike = volume_spike[i]
        wr_val = williams_r[i]
        st_dir = supertrend_direction_aligned[i]  # 1 for uptrend, -1 for downtrend
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) AND 12h Supertrend bullish AND volume spike
            if wr_val < -80 and st_dir == 1 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) AND 12h Supertrend bearish AND volume spike
            elif wr_val > -20 and st_dir == -1 and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit conditions for long
            exit_signal = False
            # Exit: Williams %R > -50 (exiting oversold territory)
            if wr_val > -50:
                exit_signal = True
            # Exit: 12h Supertrend turns bearish
            elif st_dir == -1:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit conditions for short
            exit_signal = False
            # Exit: Williams %R < -50 (exiting overbought territory)
            if wr_val < -50:
                exit_signal = True
            # Exit: 12h Supertrend turns bullish
            elif st_dir == 1:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals