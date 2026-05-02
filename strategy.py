#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA34 trend filter and volume confirmation
# Williams Alligator (Jaw/Teeth/Lips) identifies trending vs ranging markets
# Long when Lips > Teeth > Jaw + price above Alligator + 1d uptrend + volume spike
# Short when Lips < Teeth < Jaw + price below Alligator + 1d downtrend + volume spike
# Designed for 12h timeframe to capture medium-term trends in both bull and bear markets
# Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag

name = "12h_WilliamsAlligator_1dEMA34_Trend_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1d for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator on 12h timeframe
    # Jaw: 13-period SMMA smoothed 8 periods ahead
    # Teeth: 8-period SMMA smoothed 5 periods ahead  
    # Lips: 5-period SMMA smoothed 3 periods ahead
    def smma(data, period):
        """Smoothed Moving Average"""
        if len(data) < period:
            return np.full_like(data, np.nan, dtype=float)
        result = np.full_like(data, np.nan, dtype=float)
        # First value is simple SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (PREV_SMMA*(period-1) + CURRENT_DATA) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Shift to avoid look-ahead (Alligator uses future smoothing)
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    # Set rolled values to NaN
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Volume confirmation (2.0x 20-period average) on 12h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Alligator calculations)
    start_idx = 30
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Alligator alignment: Lips > Teeth > Jaw (bullish) or Lips < Teeth < Jaw (bearish)
            bullish_align = lips[i] > teeth[i] and teeth[i] > jaw[i]
            bearish_align = lips[i] < teeth[i] and teeth[i] < jaw[i]
            
            # Long entry: Bullish Alligator + price above Alligator + 1d uptrend + volume spike
            if bullish_align and close[i] > lips[i] and close[i] > ema_34_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: Bearish Alligator + price below Alligator + 1d downtrend + volume spike
            elif bearish_align and close[i] < lips[i] and close[i] < ema_34_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Bearish Alligator alignment or price below Teeth or trend reversal
            bearish_align = lips[i] < teeth[i] and teeth[i] < jaw[i]
            if bearish_align or close[i] < teeth[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Bullish Alligator alignment or price above Teeth or trend reversal
            bullish_align = lips[i] > teeth[i] and teeth[i] > jaw[i]
            if bullish_align or close[i] > teeth[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals