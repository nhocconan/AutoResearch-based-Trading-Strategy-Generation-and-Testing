#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_4hTrend_VolumeSpike_v1
Hypothesis: 4h trend filter (price vs 4h EMA50) + Camarilla R1/S1 breakouts from 1h with volume spike (>1.8x average) captures intraday momentum aligned with higher timeframe trend. Volume confirmation reduces false breakouts. Designed for 1h timeframe targeting 15-35 trades/year with discrete sizing (0.20) to minimize fee drag. Works in bull/bear via 4h trend alignment.
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
    
    # Load 4h data ONCE before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla levels from previous 1h bar (using 1h data resampled conceptually via shift)
    # For 1h timeframe, we use the previous completed 1h bar for Camarilla calculation
    high_1h = high  # we'll use shift inside loop for previous bar
    low_1h = low
    close_1h = close
    
    # ATR(14) for volatility (used in volume spike threshold)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Average volume for confirmation (24-period SMA = 1d * 1 = 1d for 1h timeframe)
    avg_volume = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    base_size = 0.20
    
    # Warmup: max of EMA(50), volume(24)
    start_idx = max(50, 24)
    
    for i in range(start_idx, n):
        close_val = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_val = ema_50_4h_aligned[i]
        
        # Get previous completed 1h bar for Camarilla levels
        if i >= 1:
            prev_high = high[i-1]
            prev_low = low[i-1]
            prev_close = close[i-1]
            
            camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
            camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
        else:
            # Not enough data for Camarilla calculation
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Skip if any data not ready
        if (np.isnan(ema_val) or np.isnan(avg_vol)):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Volume confirmation: current volume > 1.8x average volume
        volume_confirmed = vol > 1.8 * avg_vol
        
        # Trend filter: price vs 4h EMA50
        uptrend = close_val > ema_val
        downtrend = close_val < ema_val
        
        # Long: price CLOSES above R1 with 4h uptrend and volume
        long_condition = (close_val > camarilla_r1) and uptrend and volume_confirmed
        # Short: price CLOSES below S1 with 4h downtrend and volume
        short_condition = (close_val < camarilla_s1) and downtrend and volume_confirmed
        
        # Exit: price retests broken level
        long_exit = (position == 1 and close_val <= camarilla_r1)
        short_exit = (position == -1 and close_val >= camarilla_s1)
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
            entry_price = close_val
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
            entry_price = close_val
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hTrend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0