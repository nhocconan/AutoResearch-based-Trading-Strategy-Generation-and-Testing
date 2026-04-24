#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R extreme with 1d EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 4h for lower trade frequency and better signal quality.
- HTF: 1d EMA50 for trend direction (bullish if close > EMA50, bearish if close < EMA50).
- Volume: Current 4h volume > 1.8 * 20-period volume MA to capture institutional interest.
- Williams %R: Calculated on 4h data, long when < -80 (oversold), short when > -20 (overbought).
- Entry: Long when Williams %R < -80 AND 1d EMA50 bullish AND volume spike.
         Short when Williams %R > -20 AND 1d EMA50 bearish AND volume spike.
- Exit: Williams %R reverts to -50 (mean reversion) OR loss of volume confirmation.
- Signal size: 0.25 discrete to balance return and drawdown.
- Target: 80-150 total trades over 4 years (20-38/year) for 4h timeframe.
This strategy captures mean reversion from extreme momentum readings, filtered by daily trend
to avoid counter-trend trades. Volume spikes confirm institutional participation during
extremes, improving signal quality in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    df_1d_close = df_1d['close'].values
    ema_1d = pd.Series(df_1d_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 20-period 1d volume MA
    df_1d_volume = df_1d['volume'].values
    vol_ma_1d = pd.Series(df_1d_volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Williams %R on 4h data (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero (when highest_high == lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align HTF indicators to 4h
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Volume confirmation: current 4h volume > 1.8 * 20-period 1d volume MA (aligned)
    volume_spike = volume > (1.8 * vol_ma_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 14)  # Need enough bars for EMA50, volume MA, and Williams %R
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(williams_r[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_wr = williams_r[i]
        
        if position == 0:
            # Check for extreme Williams %R signals with volume spike
            if volume_spike[i]:
                # Bullish extreme: Williams %R < -80 (oversold) AND 1d EMA50 bullish (close > EMA)
                if curr_wr < -80 and curr_close > ema_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Bearish extreme: Williams %R > -20 (overbought) AND 1d EMA50 bearish (close < EMA)
                elif curr_wr > -20 and curr_close < ema_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Williams %R reverts to -50 OR loss of volume confirmation
            if curr_wr >= -50 or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R reverts to -50 OR loss of volume confirmation
            if curr_wr <= -50 or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_Extreme_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0