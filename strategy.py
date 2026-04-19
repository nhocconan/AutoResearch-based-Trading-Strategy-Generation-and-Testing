#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1d trend filter and 4h volume confirmation
# - 1d EMA(34) defines trend direction (long when price > EMA34, short when price < EMA34)
# - 4h volume > 1.5x 20-period average for confirmation
# - 12h close crosses above/below 1d EMA34 for entry
# - Exit on opposite cross or trend reversal
# - Session filter: only trade 08:00-20:00 UTC to avoid low-volume periods
# - Position size: 0.25 (25%) to balance risk and return
# - Designed to work in both bull and bear markets by following higher timeframe trend
# - Target: 15-30 trades/year to avoid excessive fee drift

name = "12h_EMA34_4hVolume_1dTrend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA(34) for trend direction
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get 4h data for volume confirmation
    df_4h = get_htf_data(prices, '4h')
    
    # 4h volume average (20-period)
    vol_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    # Pre-compute session filter (08:00-20:00 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_4h_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 4h volume > 1.5x average
        volume_filter = vol_ma_4h_aligned[i] > 0 and volume[i] > 1.5 * vol_ma_4h_aligned[i]
        
        if position == 0:
            # Look for long entry: price crosses above 1d EMA34 + volume confirmation
            if close[i] > ema_34_1d_aligned[i] and close[i-1] <= ema_34_1d_aligned[i-1] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Look for short entry: price crosses below 1d EMA34 + volume confirmation
            elif close[i] < ema_34_1d_aligned[i] and close[i-1] >= ema_34_1d_aligned[i-1] and volume_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit on cross below EMA34 or trend reversal
            if close[i] < ema_34_1d_aligned[i] or close[i-1] >= ema_34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit on cross above EMA34 or trend reversal
            if close[i] > ema_34_1d_aligned[i] or close[i-1] <= ema_34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals