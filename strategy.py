#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike_v1
Hypothesis: 1h Camarilla pivot breakout strategy with 4h trend filter and volume confirmation.
- Uses 1h timeframe for higher frequency but with tight filters to control trade count
- 4h timeframe provides signal direction (trend filter) to avoid counter-trend trades
- Camarilla R1/S1 from previous 4h bar for precise entries
- Volume spike (2x 20-period average) confirms breakout strength
- Session filter (08-20 UTC) reduces noise during low-liquidity periods
- Target: 60-150 total trades over 4 years (15-37/year) to minimize fee drag
- Works in bull/bear markets by trading with 4h trend and using Camarilla for precise entries
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC) - avoid trading during low liquidity
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for trend filter and Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate Camarilla levels from previous 4h bar
    # Camarilla: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h_arr = df_4h['close'].values
    
    camarilla_range = (high_4h - low_4h) * 1.1 / 12
    r1_4h = close_4h_arr + camarilla_range
    s1_4h = close_4h_arr - camarilla_range
    
    # Align Camarilla levels to 1h timeframe (wait for completed 4h bar)
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    
    # Calculate volume spike (20-period volume average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 2.0)  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for volume MA, 50 for EMA)
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            not in_session[i]):
            # Hold current position or go flat if outside session
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Camarilla breakout conditions with volume confirmation
        price_above_r1 = close[i] > r1_aligned[i]
        price_below_s1 = close[i] < s1_aligned[i]
        
        # 4h trend filter
        trend_up = close[i] > ema50_4h_aligned[i]
        trend_down = close[i] < ema50_4h_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 AND volume spike AND 4h uptrend AND in session
            if price_above_r1 and volume_spike[i] and trend_up:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1 AND volume spike AND 4h downtrend AND in session
            elif price_below_s1 and volume_spike[i] and trend_down:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.20
            # Exit: price falls below S1 OR 4h trend turns down OR outside session
            if price_below_s1 or not trend_up or not in_session[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            # Exit: price rises above R1 OR 4h trend turns up OR outside session
            if price_above_r1 or not trend_down or not in_session[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0