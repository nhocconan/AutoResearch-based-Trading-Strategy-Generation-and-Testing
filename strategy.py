#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R extreme with 1d trend filter and volume spike confirmation.
- Primary timeframe: 6h for balanced trade frequency and noise reduction.
- HTF: 1d EMA50 for trend direction (bullish if close > EMA50, bearish if close < EMA50).
- Williams %R: 14-period indicator for extreme oversold/overbought conditions.
- Volume: Current 6h volume > 2.0 * 24-period volume MA to capture institutional interest.
- Entry: Long when Williams %R < -80 (extreme oversold) AND 1d EMA50 bullish AND volume spike.
         Short when Williams %R > -20 (extreme overbought) AND 1d EMA50 bearish AND volume spike.
- Exit: Opposite Williams %R level (Williams %R > -20 for long, Williams %R < -80 for short) 
        OR loss of volume confirmation.
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
This strategy targets mean reversion in ranging markets while filtering for trend direction 
on higher timeframe to avoid counter-trend trades. Williams %R extremes combined with 
volume spikes identify high-probability reversal points with institutional participation.
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
    
    # Calculate 6h Williams %R(14)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        ((highest_high - close) / (highest_high - lowest_low)) * -100,
        -50.0  # neutral value when range is zero
    )
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    df_1d_close = df_1d['close'].values
    ema_1d = pd.Series(df_1d_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 24-period 1d volume MA (approximates 6d MA for volume confirmation)
    df_1d_volume = df_1d['volume'].values
    vol_ma_1d = pd.Series(df_1d_volume).rolling(window=24, min_periods=24).mean().values
    
    # Align HTF indicators to 6h
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Volume confirmation: current 6h volume > 2.0 * 24-period 1d volume MA (aligned)
    volume_spike = volume > (2.0 * vol_ma_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 24, 14)  # Need enough bars for EMA50, volume MA, and Williams %R
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(williams_r[i]) or np.isnan(volume_spike[i])):
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
                # Bullish: Williams %R < -80 (extreme oversold) AND 1d EMA50 bullish (close > EMA)
                if curr_williams_r < -80.0 and curr_close > ema_val:
                    signals[i] = 0.25
                    position = 1
                # Bearish: Williams %R > -20 (extreme overbought) AND 1d EMA50 bearish (close < EMA)
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

name = "6h_WilliamsR_Extreme_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0