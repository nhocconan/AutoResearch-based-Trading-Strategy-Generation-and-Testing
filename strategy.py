#!/usr/bin/env python3
"""
Hypothesis: 6h Volume-Weighted RSI + 1d Donchian Channel Breakout Filter.
- Primary timeframe: 6h for execution, HTF: 1d for Donchian(20) breakout direction.
- Volume-Weighted RSI(14): RSI calculated using volume-weighted price to filter low-conviction moves.
- Entry: Long when VW-RSI < 30 and price breaks above 1d Donchian upper band; Short when VW-RSI > 70 and price breaks below 1d Donchian lower band.
- Exit: Opposite VW-RSI extreme (VW-RSI > 50 for long exit, < 50 for short exit) or Donchian middle band touch.
- Volume confirmation: current volume > 1.3x 20-period volume MA to avoid low-volatility false signals.
- Discrete signal size: 0.25 to balance return and drawdown control.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
- Works in bull via buying oversold pullbacks in uptrend, in bear via selling overbought rallies in downtrend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d Donchian Channel (20-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Donchian bands: upper = highest high, lower = lowest low over 20 periods
    upper_band = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_band = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    middle_band = (upper_band + lower_band) / 2.0
    
    # Align 1d Donchian bands to 6h timeframe (completed 1d bar only)
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, lower_band)
    middle_band_aligned = align_htf_to_ltf(prices, df_1d, middle_band)
    
    # Volume-Weighted RSI(14) calculation
    # Typical Price = (High + Low + Close) / 3
    typical_price = (high + low + close) / 3.0
    # Volume-Weighted Typical Price = Typical Price * Volume
    vw_tp = typical_price * volume
    
    # Calculate changes in VWTP
    delta = vw_tp - np.roll(vw_tp, 1)
    delta[0] = 0  # First value has no previous
    
    # Separate gains and losses
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (EMA with alpha=1/14)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # RS and RSI
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    vw_rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: current volume > 1.3 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.3 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14, 20) + 1  # Donchian(20), VW-RSI(14), volume MA(20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or 
            np.isnan(middle_band_aligned[i]) or np.isnan(vw_rsi[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Entry conditions: VW-RSI extreme + Donchian breakout + volume confirmation
            if vw_rsi[i] < 30 and close[i] > upper_band_aligned[i] and volume_spike[i]:
                # Oversold + breakout above upper band = long
                signals[i] = 0.25
                position = 1
            elif vw_rsi[i] > 70 and close[i] < lower_band_aligned[i] and volume_spike[i]:
                # Overbought + breakdown below lower band = short
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: VW-RSI > 50 (momentum fading) or price touches middle band
            if vw_rsi[i] > 50 or close[i] < middle_band_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: VW-RSI < 50 (momentum fading) or price touches middle band
            if vw_rsi[i] < 50 or close[i] > middle_band_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_VolumeWeightedRSI_1dDonchian_Breakout_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0