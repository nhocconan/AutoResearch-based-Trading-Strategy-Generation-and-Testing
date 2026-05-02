#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d EMA50 trend + volume confirmation + chop regime filter
# Targets 50-150 total trades over 4 years (12-37/year) to minimize fee drag
# Williams Alligator (Jaw=13, Teeth=8, Lips=5) identifies trend: 
#   Bullish when Lips > Teeth > Jaw, Bearish when Lips < Teeth < Jaw
# 1d EMA50 determines higher timeframe trend bias
# Volume spike (2x 20-period average) confirms institutional participation
# Chop regime filter (CHOP > 61.8) avoids whipsaws in ranging markets
# Works in bull markets via trend continuation and bear markets via trend reversals
# Discrete position sizing: 0.25 (25% of capital) balances exposure and risk
# Uses 12h primary timeframe as specified in experiment #117312

name = "12h_WilliamsAlligator_1dEMA50_VolumeSpike_ChopFilter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Williams Alligator (Jaw=13, Teeth=8, Lips=5)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Alligator lines: Smoothed Moving Average (SMMA) with offset
    def smma(values, period):
        result = np.full_like(values, np.nan, dtype=float)
        if len(values) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(values[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    close_1d = df_1d['close'].values
    jaw = smma(close_1d, 13)  # Jaw (Blue) - 13-period SMMA
    teeth = smma(close_1d, 8)  # Teeth (Red) - 8-period SMMA
    lips = smma(close_1d, 5)   # Lips (Green) - 5-period SMMA
    
    # Align to 12h timeframe (wait for completed 1d bar)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Calculate 1d EMA50 trend (prior completed 1d bar's EMA)
    ema_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate 12h volume spike (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Calculate 12h Choppiness Index (CHOP) for regime filter
    # CHOP > 61.8 = ranging market (avoid breakout trades)
    # CHOP < 38.2 = trending market (favor breakout trades)
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    max_high = pd.Series(high).rolling(window=atr_period, min_periods=atr_period).max().values
    min_low = pd.Series(low).rolling(window=atr_period, min_periods=atr_period).min().values
    
    # Avoid division by zero
    sum_tr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).sum().values
    range_hl = max_high - min_low
    chop = np.where(
        (range_hl != 0) & (sum_tr != 0),
        100 * np.log10(sum_tr / range_hl) / np.log10(atr_period),
        50.0  # neutral when undefined
    )
    chop_filter = chop > 61.8  # True when ranging (avoid trades)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = max(50, 20, atr_period)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_spike[i]) or np.isnan(chop_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Only trade in trending markets (CHOP <= 61.8)
            if not chop_filter[i]:
                # Bullish Alligator: Lips > Teeth > Jaw
                bullish_alligator = (lips_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > jaw_aligned[i])
                # Bearish Alligator: Lips < Teeth < Jaw
                bearish_alligator = (lips_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < jaw_aligned[i])
                
                # Long entry: Bullish Alligator AND price > 1d EMA50 (bullish bias) AND volume spike
                if bullish_alligator and (close[i] > ema_50_aligned[i]) and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                # Short entry: Bearish Alligator AND price < 1d EMA50 (bearish bias) AND volume spike
                elif bearish_alligator and (close[i] < ema_50_aligned[i]) and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Avoid trades in ranging markets
        
        elif position == 1:  # Long position
            # Exit: Alligator turns bearish OR price falls below 1d EMA50 (trend change)
            bearish_alligator = (lips_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < jaw_aligned[i])
            if bearish_alligator or (close[i] < ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator turns bullish OR price rises above 1d EMA50 (trend change)
            bullish_alligator = (lips_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > jaw_aligned[i])
            if bullish_alligator or (close[i] > ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals