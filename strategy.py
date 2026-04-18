#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) Breakout + Weekly EMA Trend + Volume Spike
# Donchian breakouts provide clear trend signals with objective entry/exit levels.
# Weekly EMA filter ensures alignment with higher timeframe trend to avoid counter-trend trades.
# Volume spike confirms institutional participation in the breakout.
# Works in bull markets (breakouts above upper band) and bear markets (breakdowns below lower band).
# Target: 10-25 trades/year (40-100 total over 4 years) to minimize fee drag.
name = "1d_Donchian20_WeeklyEMA50_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for EMA50 trend filter
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA50
    ema_50_weekly = pd.Series(df_weekly['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_50_weekly)
    
    # Calculate Donchian channels (20-period) on daily data
    # Using rolling window with min_periods to avoid look-ahead
    high_rolling = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_rolling = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume spike: current volume > 2.5 * 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_rolling[i]) or np.isnan(low_rolling[i]) or
            np.isnan(ema_50_weekly_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        high_val = high_rolling[i]
        low_val = low_rolling[i]
        ema_val = ema_50_weekly_aligned[i]
        
        if position == 0:
            # Long: Close above upper Donchian band AND price above weekly EMA50 AND volume spike
            if close_val > high_val and close_val > ema_val and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close below lower Donchian band AND price below weekly EMA50 AND volume spike
            elif close_val < low_val and close_val < ema_val and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Close below lower Donchian band (trend reversal) or above weekly EMA50 (profit protection)
            if close_val < low_val or close_val > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Close above upper Donchian band (trend reversal) or below weekly EMA50 (profit protection)
            if close_val > high_val or close_val < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals