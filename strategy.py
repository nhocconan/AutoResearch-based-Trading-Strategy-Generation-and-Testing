#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) Breakout + Volume Spike + Weekly Trend Filter (EMA50)
# Breakout above 20-day high or below 20-day low captures strong momentum moves.
# Volume spike confirms institutional participation and reduces false breakouts.
# Weekly EMA50 filter ensures alignment with longer-term trend to avoid counter-trend trades.
# Works in bull markets (breakouts above Donchian high) and bear markets (breakdowns below Donchian low).
# Target: 10-25 trades/year (40-100 total over 4 years) to minimize fee drag.
name = "1d_Donchian20_VolumeSpike_WeeklyEMA50"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for EMA50 trend filter
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA50 for trend filter
    weekly_close = df_weekly['close'].values
    ema_50_weekly = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_50_weekly)
    
    # Calculate Donchian channels (20-period) on daily data
    # Use rolling window with min_periods to avoid look-ahead
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or
            np.isnan(ema_50_weekly_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        high_val = high_roll[i]
        low_val = low_roll[i]
        ema_val = ema_50_weekly_aligned[i]
        
        if position == 0:
            # Long: Close above Donchian high AND price above weekly EMA50 AND volume spike
            if close_val > high_val and close_val > ema_val and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close below Donchian low AND price below weekly EMA50 AND volume spike
            elif close_val < low_val and close_val < ema_val and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Close below weekly EMA50 (trend change) or at Donchian low (mean reversion)
            if close_val < ema_val or close_val < low_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Close above weekly EMA50 (trend change) or at Donchian high (mean reversion)
            if close_val > ema_val or close_val > high_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals