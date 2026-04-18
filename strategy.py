#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) Breakout + Weekly EMA10 Trend Filter + Volume Confirmation
# Breakout above 20-day high or below 20-day low with volume > 1.5x 20-day average volume.
# Weekly EMA10 filter ensures alignment with longer-term trend to avoid counter-trend trades.
# Works in bull markets (breakouts above upper band) and bear markets (breakdowns below lower band).
# Target: 10-25 trades/year (40-100 total over 4 years) to minimize fee drag.
name = "1d_Donchian20_WeeklyEMA10_Volume"
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
    
    # Get weekly data for EMA10
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA10 for trend filter
    ema_10_1w = pd.Series(df_1w['close'].values).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema_10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_10_1w)
    
    # Calculate Donchian channels (20-period) using daily data
    # Use rolling window on daily high/low
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume confirmation: current volume > 1.5x 20-day average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for Donchian calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or
            np.isnan(ema_10_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        high_max = high_max_20[i]
        low_min = low_min_20[i]
        ema_val = ema_10_1w_aligned[i]
        
        if position == 0:
            # Long: Close above 20-day high AND price above weekly EMA10 AND volume confirmation
            if close_val > high_max and close_val > ema_val and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close below 20-day low AND price below weekly EMA10 AND volume confirmation
            elif close_val < low_min and close_val < ema_val and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Close below weekly EMA10 (trend change) or at 20-day low (mean reversion)
            if close_val < ema_val or close_val < low_min:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Close above weekly EMA10 (trend change) or at 20-day high (mean reversion)
            if close_val > ema_val or close_val > high_max:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals