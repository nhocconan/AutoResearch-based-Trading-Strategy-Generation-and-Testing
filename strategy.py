#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with 1w EMA50 trend filter and volume confirmation
# Uses Williams Alligator (Jaw=13, Teeth=8, Lips=5) to identify trend absence/presence
# Only trade when Alligator lines are aligned (trending) and price breaks extreme lines
# 1w EMA50 ensures we trade only in higher timeframe trend direction
# Volume spike (1.8x 20-period average) confirms participation
# Works in bull markets via buying Teeth breaks above Lips in uptrend
# Works in bear markets via selling Teeth breaks below Lips in downtrend
# Discrete sizing 0.25 minimizes fee churn. Target: 30-100 total trades over 4 years (7-25/year).

name = "1d_Williams_Alligator_1wEMA50_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop (MTF Rule #1)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Williams Alligator: Smoothed Moving Average (SMA with specific periods)
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    def smma(source, period):
        """Smoothed Moving Average"""
        if len(source) < period:
            return np.full_like(source, np.nan, dtype=float)
        result = np.full_like(source, np.nan, dtype=float)
        # First value is simple SMA
        result[period-1] = np.mean(source[:period])
        # Subsequent values: (prev*(period-1) + current) / period
        for i in range(period, len(source)):
            result[i] = (result[i-1] * (period-1) + source[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 13  # warmup for Alligator calculation
    
    for i in range(start_idx, n):
        # Need to ensure we have valid Alligator values
        if np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_ema_50_1w = ema_50_1w_aligned[i]
        
        # Alligator alignment: Jaw > Teeth > Lips (uptrend) or Jaw < Teeth < Lips (downtrend)
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        lips_val = lips[i]
        
        # Volume confirmation: volume > 1.8x 20-period average
        vol_ma_20 = np.mean(volume[max(0, i-20):i])
        volume_spike = volume[i] > (1.8 * vol_ma_20)
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and Alligator alignment
            if volume_spike:
                # Bullish entry: Teeth crosses above Lips AND aligned uptrend
                if teeth_val > lips_val and jaw_val > teeth_val and lips_val > 0:
                    # Price breaks above Teeth with confirmation
                    if curr_close > teeth_val and curr_close > curr_ema_50_1w:
                        signals[i] = 0.25
                        position = 1
                # Bearish entry: Teeth crosses below Lips AND aligned downtrend
                elif teeth_val < lips_val and jaw_val < teeth_val and lips_val > 0:
                    # Price breaks below Teeth with confirmation
                    if curr_close < teeth_val and curr_close < curr_ema_50_1w:
                        signals[i] = -0.25
                        position = -1
        
        elif position == 1:  # Long position
            # Exit when Teeth falls below Lips or price breaks below Jaw
            if teeth_val < lips_val or curr_close < jaw_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when Teeth rises above Lips or price breaks above Jaw
            if teeth_val > lips_val or curr_close > jaw_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals