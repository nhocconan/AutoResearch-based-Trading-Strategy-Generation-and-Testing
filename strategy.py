#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h primary timeframe with 12h HTF filter
    # Strategy: Camarilla pivot breakout/mean reversion with volume confirmation
    # Long: price breaks above Camarilla R4 (12h) + volume > 1.3x 20-period avg
    # Short: price breaks below Camarilla S4 (12h) + volume > 1.3x 20-period avg
    # Mean reversion: fade at R3/S3 when price reverses back toward pivot
    # Uses discrete position sizing (0.25) to minimize fee churn
    # Target: 50-150 total trades over 4 years (12-37/year) for optimal Sharpe
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla pivot calculation (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate Camarilla levels on 12h data (using previous day's OHLC)
    # Camarilla levels: based on previous period's range
    # R4 = close + ((high - low) * 1.1/2)
    # R3 = close + ((high - low) * 1.1/4)
    # R2 = close + ((high - low) * 1.1/6)
    # R1 = close + ((high - low) * 1.1/12)
    # PP = (high + low + close) / 3
    # S1 = close - ((high - low) * 1.1/12)
    # S2 = close - ((high - low) * 1.1/6)
    # S3 = close - ((high - low) * 1.1/4)
    # S4 = close - ((high - low) * 1.1/2)
    
    # Calculate for each 12h bar using previous bar's OHLC
    prev_high = np.roll(high_12h, 1)
    prev_low = np.roll(low_12h, 1)
    prev_close = np.roll(close_12h, 1)
    
    # First bar has no previous data
    prev_high[0] = prev_low[0] = prev_close[0] = np.nan
    
    # Camarilla calculations
    camarilla_pp = (prev_high + prev_low + prev_close) / 3
    camarilla_range = prev_high - prev_low
    
    camarilla_r4 = camarilla_close + (camarilla_range * 1.1 / 2)
    camarilla_r3 = camarilla_close + (camarilla_range * 1.1 / 4)
    camarilla_s3 = camarilla_close - (camarilla_range * 1.1 / 4)
    camarilla_s4 = camarilla_close - (camarilla_range * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s4)
    
    # Volume confirmation: 20-period average on 6h data
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):  # start from 50 to have enough data for calculations
        # Skip if data not ready
        if (np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(vol_avg_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirmed = volume[i] > 1.3 * vol_avg_20[i]
        
        # Breakout conditions
        breakout_up = close[i] > camarilla_r4_aligned[i]
        breakout_down = close[i] < camarilla_s4_aligned[i]
        
        # Mean reversion conditions (fade at extremes)
        fade_long = close[i] < camarilla_s3_aligned[i] and close[i] > low[i]  # bouncing off S3
        fade_short = close[i] > camarilla_r3_aligned[i] and close[i] < high[i]  # bouncing off R3
        
        # Entry conditions
        enter_long = volume_confirmed and (breakout_up or fade_long)
        enter_short = volume_confirmed and (breakout_down or fade_short)
        
        # Exit conditions: return to pivot or opposite extreme
        exit_long = position == 1 and (close[i] < camarilla_pp_aligned[i] or close[i] > camarilla_r3_aligned[i])
        exit_short = position == -1 and (close[i] > camarilla_pp_aligned[i] or close[i] < camarilla_s3_aligned[i])
        
        # Execute signals
        if enter_long and position != 1:
            position = 1
            signals[i] = position_size
        elif enter_short and position != -1:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_12h_camarilla_breakout_meanrev_volume_v1"
timeframe = "6h"
leverage = 1.0