#!/usr/bin/env python3
"""
Hypothesis: 12h Williams %R extreme with 1d trend filter and volume confirmation.
- Primary timeframe: 12h for lower trade frequency and reduced fee drag.
- HTF: 1d EMA50 for trend direction (bullish if close > EMA50, bearish if close < EMA50).
- Williams %R: 14-period momentum oscillator measuring overbought/oversold levels.
- Volume: Current 12h volume > 1.8 * 20-period 12h volume MA to confirm participation.
- Entry: Long when Williams %R < -80 (oversold) AND 1d EMA50 bullish AND volume spike.
         Short when Williams %R > -20 (overbought) AND 1d EMA50 bearish AND volume spike.
- Exit: Opposite Williams %R level (%R > -20 for long, %R < -80 for short) or loss of volume confirmation.
- Signal size: 0.25 discrete to balance profit potential and drawdown control.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
This strategy captures mean reversions in extreme momentum while aligning with the daily trend,
avoiding counter-trend trades. Volume spikes filter for institutional interest, working in both
bull and bear markets by only taking trades in the direction of the 1d trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h Williams %R(14)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    denominator = highest_high - lowest_low
    williams_r = np.where(
        denominator != 0,
        -100.0 * (highest_high - close) / denominator,
        -50.0  # neutral when no range
    )
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    df_1d_close = df_1d['close'].values
    ema_1d = pd.Series(df_1d_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 20-period 12h volume MA
    vol_ma_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 12h
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: current 12h volume > 1.8 * 20-period 12h volume MA
    volume_spike = volume > (1.8 * vol_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 50, 20)  # Need enough bars for Williams %R, EMA50, and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(williams_r[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_williams_r = williams_r[i]
        ema_val = ema_1d_aligned[i]
        
        if position == 0:
            # Check for entry signals with volume spike
            if volume_spike[i]:
                # Bullish: Williams %R < -80 (oversold) AND 1d EMA50 bullish (close > EMA)
                if curr_williams_r < -80.0 and curr_close > ema_val:
                    signals[i] = 0.25
                    position = 1
                # Bearish: Williams %R > -20 (overbought) AND 1d EMA50 bearish (close < EMA)
                elif curr_williams_r > -20.0 and curr_close < ema_val:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Williams %R > -20 (overbought) OR loss of volume confirmation
            if curr_williams_r > -20.0 or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R < -80 (oversold) OR loss of volume confirmation
            if curr_williams_r < -80.0 or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsR_Extreme_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0