#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Williams Alligator with 1d trend filter and volume spike
    # Williams Alligator (Jaw/Teeth/Lips) identifies trending vs ranging markets
    # When Lips cross above Teeth/Jaw = uptrend, below = downtrend
    # Combined with 1d EMA50 trend filter and volume spike for institutional confirmation
    # Works in bull/bear: Alligator catches trends, volume confirms breakouts
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams Alligator on 4h data
    # Jaw: 13-period SMMA, 8-period offset
    # Teeth: 8-period SMMA, 5-period offset  
    # Lips: 5-period SMMA, 3-period offset
    def smma(series, period):
        # Smoothed Moving Average
        sma = np.full_like(series, np.nan, dtype=float)
        if len(series) >= period:
            sma[period-1] = np.mean(series[:period])
            for i in range(period, len(series)):
                sma[i] = (sma[i-1] * (period-1) + series[i]) / period
        return sma
    
    jaw_raw = smma(close, 13)
    teeth_raw = smma(close, 8)
    lips_raw = smma(close, 5)
    
    # Apply offsets
    jaw = np.roll(jaw_raw, 8)
    teeth = np.roll(teeth_raw, 5)
    lips = np.roll(lips_raw, 3)
    
    # Load 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20  # Require 2x volume for confirmation
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Lips above Teeth and Jaw (bullish alignment) + volume spike + price above 1d EMA50 (uptrend)
            if lips[i] > teeth[i] and teeth[i] > jaw[i] and vol_spike[i] and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Lips below Teeth and Jaw (bearish alignment) + volume spike + price below 1d EMA50 (downtrend)
            elif lips[i] < teeth[i] and teeth[i] < jaw[i] and vol_spike[i] and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: When Alligator lines intertwine (market ranging) or opposite alignment
            if position == 1:
                # Exit long when Lips cross below Teeth OR Teeth cross below Jaw (trend weakening)
                if lips[i] < teeth[i] or teeth[i] < jaw[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short when Lips cross above Teeth OR Teeth cross above Jaw (trend weakening)
                if lips[i] > teeth[i] or teeth[i] > jaw[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Williams_Alligator_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0