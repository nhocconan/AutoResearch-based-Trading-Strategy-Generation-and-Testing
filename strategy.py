#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout + weekly EMA trend + volume confirmation.
# Donchian(20) breakout captures momentum, weekly EMA34 filters trend direction.
# Volume spike (>1.5x 20-period average) confirms conviction.
# Works in bull markets (breakouts above upper band) and bear markets (breakdowns below lower band).
# Target: 10-25 trades/year (40-100 total over 4 years) to minimize fee drag.
name = "1d_Donchian20_WeeklyEMA34_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Donchian channels (already daily from prices)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Get weekly data for EMA trend filter
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate EMA34 on weekly data
    close_weekly = pd.Series(df_weekly['close'].values)
    ema_34_weekly = close_weekly.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA to daily timeframe
    ema_34_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_34_weekly)
    
    # Calculate volume spike: current volume > 1.5 * 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(ema_34_weekly_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        donchian_upper_val = donchian_upper[i]
        donchian_lower_val = donchian_lower[i]
        ema_val = ema_34_weekly_aligned[i]
        
        if position == 0:
            # Long: Close above upper Donchian AND price above EMA34 AND volume spike
            if close_val > donchian_upper_val and close_val > ema_val and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close below lower Donchian AND price below EMA34 AND volume spike
            elif close_val < donchian_lower_val and close_val < ema_val and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Close below EMA34 (trend change)
            if close_val < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Close above EMA34 (trend change)
            if close_val > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals