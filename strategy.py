#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_VolumeSpike_1wTrend_v1
Hypothesis: Daily Donchian(20) breakout with weekly EMA50 trend filter and volume confirmation.
- Primary timeframe: 1d targeting 30-100 total trades over 4 years (7-25/year)
- Long when price breaks above 20-day high AND weekly EMA50 uptrend AND volume spike (2.0x 20-day avg)
- Short when price breaks below 20-day low AND weekly EMA50 downtrend AND volume spike
- Donchian channels provide clear structure with proven edge in both bull and bear markets
- Weekly EMA50 filter reduces whipsaw by ensuring alignment with higher timeframe trend
- Volume spike confirms institutional participation and reduces false breakouts
- ATR-based stoploss (signal→0) manages risk in volatile conditions
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
    
    # Load weekly data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Donchian channels (20-period)
    period20_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    period20_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate weekly EMA50 for trend filter
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate volume spike (20-period volume average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 2.0)  # Volume at least 2.0x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for Donchian, 20 for volume MA)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(period20_high[i]) or np.isnan(period20_low[i]) or
            np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Donchian breakout conditions with volume confirmation and weekly trend filter
        if position == 0:
            # Long: Price breaks above 20-day high AND weekly EMA50 uptrend AND volume spike
            if close[i] > period20_high[i-1] and close[i] > ema50_1w_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 20-day low AND weekly EMA50 downtrend AND volume spike
            elif close[i] < period20_low[i-1] and close[i] < ema50_1w_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Price breaks below 20-day low OR weekly trend turns down
            if close[i] < period20_low[i-1] or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Price breaks above 20-day high OR weekly trend turns up
            if close[i] > period20_high[i-1] or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Donchian20_Breakout_VolumeSpike_1wTrend_v1"
timeframe = "1d"
leverage = 1.0