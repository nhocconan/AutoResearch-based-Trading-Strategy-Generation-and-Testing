#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolumeSpike_v1
Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and 1d volume spike.
- Uses 4h EMA50 for trend direction (bull/bear agnostic)
- Uses 1d volume spike (>1.5x 20-period average) to confirm momentum
- Enters long/short on 1h breakout above R1/below S1 only when aligned with 4h trend and volume spike
- Exits on opposite Camarilla level (S1 for longs, R1 for shorts) or trend reversal
- Target: 15-37 trades/year (60-150 over 4 years) on 1h timeframe to minimize fee drag
- Works in bull/bear markets by trading with 4h trend and using volume to filter false breakouts
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Load 1d data ONCE for volume spike filter
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_ma20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Camarilla pivot calculation (based on previous day)
    # We'll calculate daily pivots and align to 1h
    # For simplicity, we use rolling window of 24 periods (1h * 24 = 1d) to approximate
    lookback = 24  # 24 * 1h = 1 day
    if n < lookback + 1:
        return np.zeros(n)
    
    # Calculate rolling high, low, close for pseudo-daily periods
    # Using rolling window of 24 to get daily high/low/close
    roll_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    roll_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    roll_close = pd.Series(close).rolling(window=lookback, min_periods=lookback).last().values
    
    # Camarilla levels
    R1 = roll_close + (roll_high - roll_low) * 1.1 / 12
    S1 = roll_close - (roll_high - roll_low) * 1.1 / 12
    
    # Align Camarilla levels (they are based on completed pseudo-daily periods)
    # No additional alignment needed as they're already delayed by lookback period
    
    start_idx = lookback  # Need at least one full lookback period
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(vol_ma20_1d_aligned[i]) or
            np.isnan(R1[i]) or np.isnan(S1[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Volume spike condition: current 1d volume > 1.5x 20-day average
        # We need to check if we have completed a 1d period
        # For simplicity, we use the aligned 1d volume MA and check if current volume is elevated
        # In practice, we'd need actual 1d volume, but we approximate with the condition that
        # the aligned vol_ma20_1d is based on completed 1d bars
        volume_spike = volume[i] > 1.5 * vol_ma20_1d_aligned[i] if not np.isnan(vol_ma20_1d_aligned[i]) else False
        
        # 4h trend filter
        # We approximate 4h trend by checking if current price is above/below 4h EMA50
        # Since ema50_4h_aligned is already aligned to 1h and represents completed 4h bars
        uptrend = close[i] > ema50_4h_aligned[i]
        downtrend = close[i] < ema50_4h_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 AND uptrend AND volume spike
            if close[i] > R1[i] and uptrend and volume_spike:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1 AND downtrend AND volume spike
            elif close[i] < S1[i] and downtrend and volume_spike:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.20
            # Exit: price breaks below S1 OR trend reverses to downtrend
            if close[i] < S1[i] or downtrend:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            # Exit: price breaks above R1 OR trend reverses to uptrend
            if close[i] > R1[i] or uptrend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolumeSpike_v1"
timeframe = "1h"
leverage = 1.0