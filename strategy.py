#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_v2
Hypothesis: Camarilla R1/S1 breakout with 1d trend filter and volume spike. 
Long when price breaks above R1 AND 1d close > EMA(34) AND volume > 2x 20-period average.
Short when price breaks below S1 AND 1d close < EMA(34) AND volume > 2x 20-period average.
Uses discrete sizing (0.25) to limit fee churn. Target: 50-150 trades over 4 years = 12-37/year.
Works in bull (trend continuation via R1/S1 breakout) and bear (counter-trend retracement to S1/R1) via volume confirmation and trend filter.
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1d for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of lookback for Camarilla (previous day), volume MA(20)
    start_idx = 288 + 20  # 288 = 12h bars in 1 day + 20 for volume MA
    
    for i in range(start_idx, n):
        # Need previous day's high, low, close for Camarilla calculation
        if i < 288:
            # Hold current position until we have enough data
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
            
        # Get previous day's OHLC (288 bars = 1 day at 12h)
        prev_day_idx = i - 288
        if prev_day_idx < 0:
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
            
        # Previous day's high, low, close
        prev_high = high[prev_day_idx]
        prev_low = low[prev_day_idx]
        prev_close = close[prev_day_idx]
        
        # Calculate Camarilla levels for today
        range_val = prev_high - prev_low
        if range_val <= 0:
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
            
        # Camarilla R1 and S1 levels
        R1 = prev_close + (range_val * 1.1 / 12)
        S1 = prev_close - (range_val * 1.1 / 12)
        
        # Volume confirmation: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
            volume_confirm = volume[i] > (2.0 * vol_ma)
        else:
            volume_confirm = False
        
        # Trend filter from 1d EMA(34)
        if np.isnan(ema_34_1d_aligned[i]):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
            
        regime_long = close[i] > ema_34_1d_aligned[i]  # 1d uptrend
        regime_short = close[i] < ema_34_1d_aligned[i]  # 1d downtrend
        
        current_price = close[i]
        
        if position == 0:
            # Long: price breaks above R1 AND volume confirm AND 1d uptrend
            long_signal = (current_price > R1) and volume_confirm and regime_long
            
            # Short: price breaks below S1 AND volume confirm AND 1d downtrend
            short_signal = (current_price < S1) and volume_confirm and regime_short
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price breaks below S1 (reversal) OR 1d trend flips down
            if (current_price < S1) or (not regime_long):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price breaks above R1 (reversal) OR 1d trend flips up
            if (current_price > R1) or (not regime_short):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_v2"
timeframe = "12h"
leverage = 1.0