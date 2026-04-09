#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Williams Alligator with Elder Ray confirmation
# Williams Alligator (Jaw/Teeth/Lips) from 1d provides trend direction aligned with 4h timeframe
# Elder Ray (Bull Power/Bear Power) confirms trend strength and filters weak moves
# Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years)
# Works in bull/bear: Alligator identifies trend, Elder Ray confirms momentum
# Discrete position sizing: 0.0, ±0.30 to minimize fee churn

name = "4h_1d_alligator_elder_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Alligator (1d)
    # Jaw: 13-period SMMA smoothed 8 periods ahead
    # Teeth: 8-period SMMA smoothed 5 periods ahead  
    # Lips: 5-period SMMA smoothed 3 periods ahead
    def smma(values, period):
        """Smoothed Moving Average"""
        if len(values) < period:
            return np.full(len(values), np.nan)
        result = np.full(len(values), np.nan)
        sma = np.mean(values[:period])
        result[period-1] = sma
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    jaw_1d = smma(smma(close_1d, 13), 8)
    teeth_1d = smma(smma(close_1d, 8), 5)
    lips_1d = smma(smma(close_1d, 5), 3)
    
    # Elder Ray (1d)
    # Bull Power = High - EMA13
    # Bear Power = Low - EMA13
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = low_1d - ema13_1d
    
    # Align Alligator lines and Elder Ray to 4h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_1d)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_1d)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_1d)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Pre-compute volume confirmation (20-period average for 4h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i]) or np.isnan(bull_power_aligned[i]) or
            np.isnan(bear_power_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x average 4h volume
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 1:  # Long position
            # Exit when Lips cross below Teeth (trend weakening) OR Bear Power > 0 (bulls losing)
            if lips_aligned[i] < teeth_aligned[i] or bear_power_aligned[i] > 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit when Lips cross above Teeth (trend weakening) OR Bull Power < 0 (bears losing)
            if lips_aligned[i] > teeth_aligned[i] or bull_power_aligned[i] < 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat
            # Trend following with volume confirmation
            # Long when Lips > Teeth > Jaw (bullish alignment) AND Bull Power > 0
            # Short when Lips < Teeth < Jaw (bearish alignment) AND Bear Power < 0
            if volume_confirmed:
                bullish_aligned = lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]
                bearish_aligned = lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i]
                
                if bullish_aligned and bull_power_aligned[i] > 0:
                    position = 1
                    signals[i] = 0.30
                elif bearish_aligned and bear_power_aligned[i] < 0:
                    position = -1
                    signals[i] = -0.30
    
    return signals