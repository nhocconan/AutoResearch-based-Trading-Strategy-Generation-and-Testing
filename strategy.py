#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator breakout with weekly EMA8 trend filter and volume confirmation
# Williams Alligator (jaw=13, teeth=8, lips=5) identifies trend initiation when lines diverge.
# Weekly EMA8 ensures we trade with the primary trend. Volume spike confirms institutional participation.
# Works in bull (Alligator waking up with volume) and bear (trend continuation after pullbacks).
# Discrete sizing (0.25) minimizes fee churn. Target: 50-150 total trades over 4 years (12-37/year).

name = "12h_WilliamsAlligator_Breakout_WeeklyEMA8_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly HTF data for EMA8 calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 8:
        return np.zeros(n)
    
    # Weekly EMA(8) calculation
    close_1w = df_1w['close'].values
    ema_8_1w = pd.Series(close_1w).ewm(span=8, adjust=False, min_periods=8).mean().values
    ema_8_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_8_1w)
    
    # Williams Alligator calculation (using 5m data approximated via 12h - using close for simplicity)
    # Jaw: 13-period SMMA (smoothed moving average) of median price
    # Teeth: 8-period SMMA of median price  
    # Lips: 5-period SMMA of median price
    median_price = (high + low) / 2.0
    
    # SMMA calculation (similar to Wilder's smoothing)
    def smma(values, period):
        result = np.full_like(values, np.nan)
        if len(values) < period:
            return result
        # First value is simple SMA
        result[period-1] = np.mean(values[:period])
        # Subsequent values: SMMA = (Prev SMMA * (Period-1) + Current Value) / Period
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    jaw = smma(median_price, 13)
    teeth = smma(median_price, 8)
    lips = smma(median_price, 5)
    
    # Align Alligator lines to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, prices, jaw)  # Same timeframe, no alignment needed
    teeth_aligned = align_htf_to_ltf(prices, prices, teeth)
    lips_aligned = align_htf_to_ltf(prices, prices, lips)
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(13, 20)  # Need 13 for jaw SMMA + 20 for volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema_8_1w_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Williams Alligator breakout conditions
        # Alligator waking up: lips > teeth > jaw (bullish) or lips < teeth < jaw (bearish)
        bullish_alligator = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
        bearish_alligator = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]
        
        # Volume confirmation and trend filter
        vol_spike = volume_spike[i]
        # Trend filter: price above/below weekly EMA8
        uptrend = curr_close > ema_8_1w_aligned[i]
        downtrend = curr_close < ema_8_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Bullish Alligator alignment, volume spike, uptrend
            if bullish_alligator and vol_spike and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator alignment, volume spike, downtrend
            elif bearish_alligator and vol_spike and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on Alligator convergence or trend reversal
            if lips_aligned[i] <= teeth_aligned[i] or curr_close < ema_8_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on Alligator convergence or trend reversal
            if lips_aligned[i] >= teeth_aligned[i] or curr_close > ema_8_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals