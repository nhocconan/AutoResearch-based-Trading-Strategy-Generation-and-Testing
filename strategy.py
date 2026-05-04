#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator + Elder Ray + Volume Spike for trend following
# Uses 1d timeframe with Williams Alligator (Jaw/Teeth/Lips) for trend direction
# Elder Ray (Bull/Bear Power) for momentum confirmation
# Volume spike (2.0x 20-period EMA) to ensure strong participation
# Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag on 1d timeframe
# Works in both bull and bear markets by following the Alligator alignment and Elder Ray signals
# Prioritizes BTC/ETH performance with SOL as secondary

name = "1d_WilliamsAlligator_ElderRay_Volume_Spike"
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
    
    # Get 1w data for HTF indicators (as per experiment instruction)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 13:
        return np.zeros(n)
    
    # Calculate 1w EMA13 for HTF trend filter
    close_1w = df_1w['close'].values
    ema_13_1w = pd.Series(close_1w).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_13_1w)
    
    # Williams Alligator on 1d timeframe
    # Jaw (blue line): 13-period SMMA shifted 8 bars
    # Teeth (red line): 8-period SMMA shifted 5 bars
    # Lips (green line): 5-period SMMA shifted 3 bars
    # SMMA = Smoothed Moving Average (similar to Wilder's smoothing)
    
    def smma(arr, period):
        """Smoothed Moving Average"""
        if len(arr) < period:
            return np.full(len(arr), np.nan)
        result = np.full(len(arr), np.nan)
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV SMMA * (PERIOD-1) + CLOSE) / PERIOD
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)  # 13-period SMMA
    teeth = smma(close, 8)  # 8-period SMMA
    lips = smma(close, 5)   # 5-period SMMA
    
    # Shift the lines as per Alligator definition
    jaw_shifted = np.full_like(jaw, np.nan)
    teeth_shifted = np.full_like(teeth, np.nan)
    lips_shifted = np.full_like(lips, np.nan)
    
    if len(jaw) > 8:
        jaw_shifted[8:] = jaw[:-8]
    if len(teeth) > 5:
        teeth_shifted[5:] = teeth[:-5]
    if len(lips) > 3:
        lips_shifted[3:] = lips[:-3]
    
    # Elder Ray on 1d timeframe
    # Bull Power = High - EMA13
    # Bear Power = Low - EMA13
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volume confirmation: 2.0x 20-period EMA
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start from 50 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(ema_13_1w_aligned[i]) or
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator alignment: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        alligator_long = lips_shifted[i] > teeth_shifted[i] and teeth_shifted[i] > jaw_shifted[i]
        alligator_short = lips_shifted[i] < teeth_shifted[i] and teeth_shifted[i] < jaw_shifted[i]
        
        # Elder Ray confirmation: Bull Power > 0 and rising, Bear Power < 0 and falling
        # Simplified: Bull Power > 0 for long, Bear Power < 0 for short
        elder_long = bull_power[i] > 0
        elder_short = bear_power[i] < 0
        
        # HTF trend filter: price above/below 1w EMA13
        htf_uptrend = close[i] > ema_13_1w_aligned[i]
        htf_downtrend = close[i] < ema_13_1w_aligned[i]
        
        # Entry conditions
        if position == 0:
            # Long: Alligator aligned up + Elder Ray bullish + HTF uptrend + Volume spike
            if (alligator_long and elder_long and htf_uptrend and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Alligator aligned down + Elder Ray bearish + HTF downtrend + Volume spike
            elif (alligator_short and elder_short and htf_downtrend and volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator alignment breaks down OR Elder Ray turns bearish
            if not alligator_long or not elder_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator alignment breaks up OR Elder Ray turns bullish
            if not alligator_short or not elder_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals