#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike_v1
Hypothesis: 1h Camarilla(R1/S1) breakout with 4h EMA34 trend filter and volume spike confirmation.
- Long when price breaks above Camarilla R1 AND 4h EMA34 uptrend AND volume > 1.8 * volume_ma(20)
- Short when price breaks below Camarilla S1 AND 4h EMA34 downtrend AND volume > 1.8 * volume_ma(20)
- Uses Camarilla pivot levels from completed 1h bar (based on prior 4h bar) for intraday structure
- 4h EMA34 filter ensures trading with higher timeframe trend to avoid counter-trend whipsaws
- Volume spike confirms institutional participation and reduces false breakouts
- Session filter (08-20 UTC) reduces noise during low-liquidity periods
- Target: 15-37 trades/year on 1h to minimize fee drag while capturing BTC/ETH moves in bull/bear markets
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
    open_time = prices['open_time']
    
    # Pre-compute session hours (08-20 UTC) for filtering
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for Camarilla calculation (structure)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate Camarilla levels from prior completed 4h bar
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # where C = close, H = high, L = low of prior 4h bar
    lookback = 1  # Use completed 4h bar only
    h_4h = pd.Series(df_4h['high'].values).rolling(window=lookback, min_periods=lookback).max().values
    l_4h = pd.Series(df_4h['low'].values).rolling(window=lookback, min_periods=lookback).min().values
    c_4h = pd.Series(df_4h['close'].values).rolling(window=lookback, min_periods=lookback).last().values
    
    camarilla_range = h_4h - l_4h
    r1 = c_4h + (camarilla_range * 1.1 / 12)
    s1 = c_4h - (camarilla_range * 1.1 / 12)
    
    # Align Camarilla levels to 1h timeframe (available after 4h bar closes)
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1)
    
    # Load 4h data ONCE before loop for EMA34 trend filter (HTF)
    ema_34_4h = pd.Series(df_4h['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    # Trend: 1 = uptrend (close > EMA34), -1 = downtrend (close < EMA34), 0 = neutral/invalid
    trend_4h = np.where(ema_34_4h_aligned > 0, 
                        np.where(close > ema_34_4h_aligned, 1, -1), 
                        0)
    
    # Calculate volume filter: volume > 1.8 * volume_ma(20) for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for EMA, 20 for volume MA)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(trend_4h[i]) or np.isnan(volume_ma[i]) or
            not in_session.iloc[i]):
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
            # Long: Price breaks above R1 AND 4h uptrend AND volume spike
            if close[i] > r1_aligned[i] and trend_4h[i] == 1 and volume_spike[i]:
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below S1 AND 4h downtrend AND volume spike
            elif close[i] < s1_aligned[i] and trend_4h[i] == -1 and volume_spike[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.20
            # Exit: Price falls below S1 OR 4h trend turns down
            if close[i] < s1_aligned[i] or trend_4h[i] == -1:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            # Exit: Price rises above R1 OR 4h trend turns up
            if close[i] > r1_aligned[i] or trend_4h[i] == 1:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0