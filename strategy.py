#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy combining weekly trend filter with daily volume confirmation
# - Weekly EMA(20) defines long-term trend direction
# - Daily price crossing above/below weekly EMA triggers position
# - Daily volume > 1.5x 20-day average confirms conviction
# - Position size: 0.25 (25%) to manage drawdown
# - Designed for low frequency (~10-20 trades/year) to minimize fee drag
# - Works in both bull and bear markets by following higher timeframe trend

name = "1d_WeeklyEMA20_Volume_Confirmation_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter (once before loop)
    df_weekly = get_htf_data(prices, '1w')
    
    # Weekly EMA(20) for trend direction
    ema_20_weekly = pd.Series(df_weekly['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_20_weekly)
    
    # Daily volume average (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema_20_weekly_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current daily volume > 1.5x 20-day average
        volume_filter = vol_ma_20[i] > 0 and volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Look for long entry: price above weekly EMA + volume confirmation
            if close[i] > ema_20_weekly_aligned[i] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Look for short entry: price below weekly EMA + volume confirmation
            elif close[i] < ema_20_weekly_aligned[i] and volume_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price crosses below weekly EMA
            if close[i] < ema_20_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price crosses above weekly EMA
            if close[i] > ema_20_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals