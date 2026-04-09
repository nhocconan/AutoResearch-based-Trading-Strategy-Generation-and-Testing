#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with 1w trend filter and volume confirmation
# Williams Alligator: Jaw (13-period SMMA shifted 8), Teeth (8-period SMMA shifted 5), Lips (5-period SMMA shifted 3)
# In bull/bear markets: Alligator lines aligned (Jaw > Teeth > Lips for uptrend, reverse for downtrend) + volume confirmation
# In ranging markets: Alligator lines intertwined -> no trades (avoids whipsaws)
# Uses discrete position sizing 0.25 to limit trades and reduce fee drag
# Works in bull/bear markets: trend following with Alligator avoids false signals in chop

name = "1d_1w_williams_alligator_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate Williams Alligator on 1d
    # SMMA (Smoothed Moving Average) = EMA with alpha = 1/period
    def smma(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        # First value is SMA
        result = np.full(len(values), np.nan)
        result[period-1] = np.mean(values[:period])
        # Subsequent values: SMMA[i] = (SMMA[i-1] * (period-1) + values[i]) / period
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    # Jaw: 13-period SMMA shifted 8 bars
    jaw = smma(close, 13)
    jaw_shifted = np.roll(jaw, 8)
    jaw_shifted[:8] = np.nan
    
    # Teeth: 8-period SMMA shifted 5 bars
    teeth = smma(close, 8)
    teeth_shifted = np.roll(teeth, 5)
    teeth_shifted[:5] = np.nan
    
    # Lips: 5-period SMMA shifted 3 bars
    lips = smma(close, 5)
    lips_shifted = np.roll(lips, 3)
    lips_shifted[:3] = np.nan
    
    # Calculate 1w EMA(34) for trend filter
    def ema(values, span):
        if len(values) < span:
            return np.full(len(values), np.nan)
        alpha = 2.0 / (span + 1)
        result = np.full(len(values), np.nan)
        result[0] = values[0]
        for i in range(1, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    ema_34_1w = ema(close_1w, 34)
    
    # Calculate 1d average volume (20-period) for confirmation
    volume_s = pd.Series(volume)
    avg_volume = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Align 1w indicators to 1d timeframe
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average volume
        volume_confirmed = volume[i] > 1.5 * avg_volume[i]
        
        # 1w trend filter: price above/below EMA34
        uptrend_1w = close[i] > ema_34_1w_aligned[i]
        downtrend_1w = close[i] < ema_34_1w_aligned[i]
        
        # Alligator alignment: Jaw > Teeth > Lips (uptrend) or Jaw < Teeth < Lips (downtrend)
        alligator_long = jaw_shifted[i] > teeth_shifted[i] and teeth_shifted[i] > lips_shifted[i]
        alligator_short = jaw_shifted[i] < teeth_shifted[i] and teeth_shifted[i] < lips_shifted[i]
        
        if position == 1:  # Long position
            # Exit if Alligator loses alignment or trend turns down
            if not alligator_long or not uptrend_1w:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit if Alligator loses alignment or trend turns up
            if not alligator_short or not downtrend_1w:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long if Alligator aligned for uptrend and 1w uptrend with volume
            if alligator_long and uptrend_1w and volume_confirmed:
                position = 1
                signals[i] = 0.25
            # Enter short if Alligator aligned for downtrend and 1w downtrend with volume
            elif alligator_short and downtrend_1w and volume_confirmed:
                position = -1
                signals[i] = -0.25
    
    return signals