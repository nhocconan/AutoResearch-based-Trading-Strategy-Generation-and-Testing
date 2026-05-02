#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h trend filter (EMA50) and 1d volume spike confirmation
# Uses 1h timeframe for signal generation with Camarilla levels from 4h pivots
# 4h EMA50 provides higher timeframe trend filter to avoid counter-trend trades
# 1d volume confirmation (2.0x 20-period average) ensures institutional participation
# Session filter (08-20 UTC) reduces noise trades outside active market hours
# Discrete position sizing (0.20) balances return and risk while minimizing fee churn
# Target: 60-150 total trades over 4 years = 15-37/year for 1h timeframe
# Works in bull markets via trend-aligned breakouts, in bear via trend filter avoiding false signals
# Based on proven Camarilla pivot structure with volume confirmation edge

name = "1h_Camarilla_R3_S3_Breakout_4hEMA50_1dVolumeSpike_Session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for Camarilla pivot and trend calculations
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 4h Camarilla pivot levels (based on previous 4h bar's OHLC)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h_prev = df_4h['close'].values
    
    # Shift by 1 to use previous 4h bar's data for current 4h bar's levels
    high_4h_prev = np.concatenate([[np.nan], high_4h[:-1]])
    low_4h_prev = np.concatenate([[np.nan], low_4h[:-1]])
    close_4h_prev = np.concatenate([[np.nan], close_4h[:-1]])
    
    # Calculate Camarilla levels for each 4h bar
    camarilla_r3 = close_4h_prev + (high_4h_prev - low_4h_prev) * 1.1 / 4
    camarilla_s3 = close_4h_prev - (high_4h_prev - low_4h_prev) * 1.1 / 4
    camarilla_r4 = close_4h_prev + (high_4h_prev - low_4h_prev) * 1.1 / 2
    camarilla_s4 = close_4h_prev - (high_4h_prev - low_4h_prev) * 1.1 / 2
    
    # Align Camarilla levels to 1h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s4)
    
    # Load 1d data ONCE before loop for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Volume confirmation (2.0x 20-period average)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm_1d = vol_1d > (vol_ma_1d * 2.0)
    volume_confirm_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Check for NaN values in indicators
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(volume_confirm_1d_aligned[i]) or
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above Camarilla R3 + price > 4h EMA50 + 1d volume confirm
            if close[i] > camarilla_r3_aligned[i] and close[i] > ema_50_4h_aligned[i] and volume_confirm_1d_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below Camarilla S3 + price < 4h EMA50 + 1d volume confirm
            elif close[i] < camarilla_s3_aligned[i] and close[i] < ema_50_4h_aligned[i] and volume_confirm_1d_aligned[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below Camarilla S3 or reverse signal
            if close[i] < camarilla_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: Price breaks above Camarilla R3 or reverse signal
            if close[i] > camarilla_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals