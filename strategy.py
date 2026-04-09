#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator + Elder Ray + Volume Spike
# - Uses 1d Williams Alligator (Jaw/Teeth/Lips) for trend direction
# - Confirms with 1d Elder Ray Bull/Bear Power for momentum strength
# - Filters with 1d volume > 2.0x 20-period average for institutional participation
# - Enters long when Lips > Teeth > Jaw AND Bull Power > 0 AND volume spike
# - Enters short when Lips < Teeth < Jaw AND Bear Power < 0 AND volume spike
# - Exits on opposite Alligator alignment or ATR-based stoploss (2.0x ATR)
# - Position size: 0.25 (25% of capital) for low fee drag and controlled risk
# - Target: 15-35 trades/year on 4h timeframe (60-140 total over 4 years)
# - Williams Alligator identifies trending vs ranging markets inherently
# - Elder Ray measures bull/bear power behind price moves
# - Volume spike ensures moves have conviction
# - Works in bull markets (strong uptrends with buying pressure) and bear markets (strong downtrends with selling pressure)

name = "4h_1d_williams_elderray_volume_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Pre-compute HTF indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d Williams Alligator (Smoothed Medians)
    # Jaw: 13-period SMMA smoothed by 8 periods
    # Teeth: 8-period SMMA smoothed by 5 periods
    # Lips: 5-period SMMA smoothed by 3 periods
    def smma(arr, period):
        """Smoothed Moving Average"""
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=float)
        result = np.full_like(arr, np.nan, dtype=float)
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CLOSE) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close_1d, 13)  # 13-period
    jaw = smma(jaw, 8)        # smoothed by 8
    teeth = smma(close_1d, 8)  # 8-period
    teeth = smma(teeth, 5)    # smoothed by 5
    lips = smma(close_1d, 5)   # 5-period
    lips = smma(lips, 3)      # smoothed by 3
    
    # 1d Elder Ray Index
    # Bull Power = High - EMA(13)
    # Bear Power = Low - EMA(13)
    ema_13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_1d - ema_13
    bear_power = low_1d - ema_13
    
    # 1d Volume > 2.0x 20-period average (strict for fewer trades)
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (2.0 * avg_volume_20)
    
    # 1d ATR(14) for stoploss
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align all 1d indicators to 4h
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 4h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    atr_stop = 0.0
    
    for i in range(40, n):
        # Skip if any required data is invalid
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(volume_spike_aligned[i]) or np.isnan(atr_1d_aligned[i]) or
            atr_1d_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions: opposite Alligator alignment OR ATR stoploss
            if lips_aligned[i] <= teeth_aligned[i] or teeth_aligned[i] <= jaw_aligned[i]:  # Lips <= Teeth OR Teeth <= Jaw
                position = 0
                signals[i] = 0.0
            elif high[i] >= entry_price + (2.0 * atr_stop):  # ATR stoploss
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: opposite Alligator alignment OR ATR stoploss
            if lips_aligned[i] >= teeth_aligned[i] or teeth_aligned[i] >= jaw_aligned[i]:  # Lips >= Teeth OR Teeth >= Jaw
                position = 0
                signals[i] = 0.0
            elif low[i] <= entry_price - (2.0 * atr_stop):  # ATR stoploss
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Alligator alignment with Elder Ray confirmation and volume spike
            # Bullish: Lips > Teeth > Jaw AND Bull Power > 0 AND volume spike
            if (lips_aligned[i] > teeth_aligned[i] and 
                teeth_aligned[i] > jaw_aligned[i] and
                bull_power_aligned[i] > 0 and 
                volume_spike_aligned[i]):
                position = 1
                entry_price = high[i]
                atr_stop = atr_1d_aligned[i]
                signals[i] = 0.25
            # Bearish: Lips < Teeth < Jaw AND Bear Power < 0 AND volume spike
            elif (lips_aligned[i] < teeth_aligned[i] and 
                  teeth_aligned[i] < jaw_aligned[i] and
                  bear_power_aligned[i] < 0 and 
                  volume_spike_aligned[i]):
                position = -1
                entry_price = low[i]
                atr_stop = atr_1d_aligned[i]
                signals[i] = -0.25
    
    return signals