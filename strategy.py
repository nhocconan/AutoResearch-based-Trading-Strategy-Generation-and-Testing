#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike_v1
Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA34 trend filter and volume spike confirmation.
- Long when price breaks above Camarilla R1 AND 4h EMA34 uptrend AND volume > 1.8 * volume_ma(20)
- Short when price breaks below Camarilla S1 AND 4h EMA34 downtrend AND volume > 1.8 * volume_ma(20)
- Uses Camarilla pivots from prior 1h bar (structure) and 4h EMA34 for trend alignment
- Volume spike confirms institutional participation and reduces false breakouts
- Session filter (08-20 UTC) to avoid low-liquidity periods
- Discrete position size 0.20 to minimize fee churn
- Designed for 1h timeframe targeting 15-35 trades/year (60-140 over 4 years)
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
    
    # Load 1h data ONCE before loop for Camarilla pivots (structure)
    df_1h = get_htf_data(prices, '1h')
    
    # Calculate Camarilla levels from prior 1h bar (completed bar only)
    # Camarilla R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    camarilla_high = df_1h['close'].values + (1.1/12) * (df_1h['high'].values - df_1h['low'].values)
    camarilla_low = df_1h['close'].values - (1.1/12) * (df_1h['high'].values - df_1h['low'].values)
    
    # Align Camarilla levels to 1h timeframe (no additional delay needed for structure)
    camarilla_high_aligned = align_htf_to_ltf(prices, df_1h, camarilla_high)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_1h, camarilla_low)
    
    # Load 4h data ONCE before loop for trend filter (HTF)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA34 for trend filter (needs completed 4h candle)
    ema_34_4h = pd.Series(df_4h['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    # Trend: 1 = uptrend (close > EMA34), -1 = downtrend (close < EMA34), 0 = neutral/invalid
    trend_4h = np.where(ema_34_4h_aligned > 0, 
                        np.where(close > ema_34_4h_aligned, 1, -1), 
                        0)
    
    # Calculate volume filter: volume > 1.8 * volume_ma(20) for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * volume_ma)
    
    # Session filter: 08-20 UTC (avoid low-liquidity periods)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for EMA, 20 for volume MA)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(camarilla_high_aligned[i]) or np.isnan(camarilla_low_aligned[i]) or
            np.isnan(trend_4h[i]) or np.isnan(volume_ma[i]) or not session_filter[i]):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Camarilla breakout conditions with trend and volume spike filter
        if position == 0:
            # Long: Price breaks above Camarilla R1 AND 4h uptrend AND volume spike
            if close[i] > camarilla_high_aligned[i] and trend_4h[i] == 1 and volume_spike[i]:
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below Camarilla S1 AND 4h downtrend AND volume spike
            elif close[i] < camarilla_low_aligned[i] and trend_4h[i] == -1 and volume_spike[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.20
            # Exit: Price falls below Camarilla S1 OR 4h trend turns down
            if close[i] < camarilla_low_aligned[i] or trend_4h[i] == -1:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            # Exit: Price rises above Camarilla R1 OR 4h trend turns up
            if close[i] > camarilla_high_aligned[i] or trend_4h[i] == 1:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0