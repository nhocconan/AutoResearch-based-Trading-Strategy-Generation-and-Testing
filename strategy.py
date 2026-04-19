#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1w trend filter and volume confirmation
# - Williams Alligator (13,8,5 SMAs on median price) defines trend: Jaw(13), Teeth(8), Lips(5)
# - Trend up when Lips > Teeth > Jaw, down when Lips < Teeth < Jaw
# - 1w trend filter: only trade in direction of weekly EMA(20) for higher timeframe bias
# - Volume confirmation: current 6h volume > 1.5x 20-period average for conviction
# - Entry on Alligator alignment in trend direction with volume
# - Exit when Alligator lines intertwine (loss of trend structure) or opposite alignment
# - Session filter: 08:00-20:00 UTC to avoid low-volume periods
# - Position size: 0.25 to manage drawdown in volatile 6h timeframe
# - Designed to capture strong trends while avoiding whipsaws in ranging markets
# - Target: 20-40 trades/year to stay within fee drag limits

name = "6h_WilliamsAlligator_1wTrend_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # 1w EMA(20) for trend direction
    ema_20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Williams Alligator components (13,8,5 SMAs on median price)
    median_price = (high + low) / 2
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values  # 13-period
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values    # 8-period
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values     # 5-period
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    
    # 1d volume average (20-period) aligned to 6s
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Pre-compute session filter (08:00-20:00 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for Alligator and volume
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 6h volume > 1.5x 1d average
        volume_filter = vol_ma_1d_aligned[i] > 0 and volume[i] > 1.5 * vol_ma_1d_aligned[i]
        
        # Alligator alignment checks
        lips_above_teeth = lips[i] > teeth[i]
        teeth_above_jaw = teeth[i] > jaw[i]
        lips_below_teeth = lips[i] < teeth[i]
        teeth_below_jaw = teeth[i] < jaw[i]
        
        # Trend up: Lips > Teeth > Jaw
        alligator_up = lips_above_teeth and teeth_above_jaw
        # Trend down: Lips < Teeth < Jaw
        alligator_down = lips_below_teeth and teeth_below_jaw
        
        if position == 0:
            # Look for long entry: uptrend alignment + weekly trend up + volume
            if alligator_up and close[i] > ema_20_1w_aligned[i] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Look for short entry: downtrend alignment + weekly trend down + volume
            elif alligator_down and close[i] < ema_20_1w_aligned[i] and volume_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when Alligator loses alignment or weekly trend fails
            if not alligator_up or close[i] < ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when Alligator loses alignment or weekly trend fails
            if not alligator_down or close[i] > ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals