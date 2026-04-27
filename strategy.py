#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_12hTrend_VolumeSpike
Hypothesis: Uses 4h Donchian channel breakout with 12h EMA50 trend filter and volume spike confirmation.
Enter long when price breaks above 20-period Donchian high AND 12h close > EMA50 (uptrend) AND volume > 2.0 * 20-period average.
Enter short when price breaks below 20-period Donchian low AND 12h close < EMA50 (downtrend) AND volume > 2.0 * 20-period average.
Exit when price returns to the opposite Donchian level (mean reversion) OR trend reverses.
Donchian channels provide clear breakout levels that work in both bull and bear markets by capturing volatility expansion.
Combined with 12h trend filter and volume confirmation, this should avoid false breakouts and work across regimes.
Target: 100-180 total trades over 4 years (25-45/year) with 0.30 position size to balance profit and drawdown.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # 12h EMA50 for trend filter
    close_12h_series = pd.Series(df_12h['close'].values)
    ema_50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 4h Donchian channel (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.30   # Position size: 30% of capital
    
    # Warmup: need 12h EMA50 (50), Donchian (20), volume avg (20)
    start_idx = max(50, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        ema_val = ema_50_12h_aligned[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Look for entry: breakout of Donchian levels with 12h trend filter AND volume
            # Long: price breaks above Donchian high AND 12h uptrend AND volume
            long_condition = (close_val > upper) and (close_val > ema_val) and vol_conf
            # Short: price breaks below Donchian low AND 12h downtrend AND volume
            short_condition = (close_val < lower) and (close_val < ema_val) and vol_conf
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long when price returns to Donchian low OR trend breaks
            exit_condition = (close_val <= lower) or (close_val < ema_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short when price returns to Donchian high OR trend breaks
            exit_condition = (close_val >= upper) or (close_val > ema_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_Breakout_12hTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0