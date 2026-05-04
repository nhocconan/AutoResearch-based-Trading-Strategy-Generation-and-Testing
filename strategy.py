#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d EMA34 trend filter + volume confirmation
# Williams Alligator consists of three SMAs (Jaw=13, Teeth=8, Lips=5) with future shifts
# Long when Lips > Teeth > Jaw (bullish alignment) AND price > Teeth AND 1d EMA34 uptrend AND volume spike
# Short when Lips < Teeth < Jaw (bearish alignment) AND price < Teeth AND 1d EMA34 downtrend AND volume spike
# Designed for 12-37 trades/year on 6h to minimize fee drag while capturing trends in both bull and bear markets.
# Alligator's smoothed nature reduces whipsaw vs pure MA crossovers.

name = "6h_WilliamsAlligator_1dEMA34_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams Alligator on 6h data
    # Jaw: 13-period SMMA shifted 8 bars forward
    # Teeth: 8-period SMMA shifted 5 bars forward  
    # Lips: 5-period SMMA shifted 3 bars forward
    # SMMA (Smoothed Moving Average) approximation using EMA with alpha=1/period
    close_series = pd.Series(close)
    
    # Jaw (13-period)
    jaw_raw = close_series.ewm(alpha=1/13, adjust=False).mean().values
    jaw = np.roll(jaw_raw, 8)  # shift forward 8 bars
    jaw[:8] = np.nan  # first 8 values invalid
    
    # Teeth (8-period)
    teeth_raw = close_series.ewm(alpha=1/8, adjust=False).mean().values
    teeth = np.roll(teeth_raw, 5)  # shift forward 5 bars
    teeth[:5] = np.nan  # first 5 values invalid
    
    # Lips (5-period)
    lips_raw = close_series.ewm(alpha=1/5, adjust=False).mean().values
    lips = np.roll(lips_raw, 3)  # shift forward 3 bars
    lips[:3] = np.nan  # first 3 values invalid
    
    # Calculate volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 2.0)  # Volume at least 2x average for confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Lips > Teeth > Jaw (bullish alignment) AND price > Teeth AND 1d uptrend AND volume spike
            if (lips[i] > teeth[i] and teeth[i] > jaw[i] and  # bullish alignment
                close[i] > teeth[i] and  # price above teeth
                close[i] > ema_34_aligned[i] and  # 1d uptrend
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Lips < Teeth < Jaw (bearish alignment) AND price < Teeth AND 1d downtrend AND volume spike
            elif (lips[i] < teeth[i] and teeth[i] < jaw[i] and  # bearish alignment
                  close[i] < teeth[i] and  # price below teeth
                  close[i] < ema_34_aligned[i] and  # 1d downtrend
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: bearish alignment OR price closes below teeth OR 1d trend turns down
            if (lips[i] < teeth[i] or teeth[i] < jaw[i] or  # bearish alignment
                close[i] < teeth[i] or  # price below teeth
                close[i] < ema_34_aligned[i]):  # 1d downtrend
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: bullish alignment OR price closes above teeth OR 1d trend turns up
            if (lips[i] > teeth[i] or teeth[i] > jaw[i] or  # bullish alignment
                close[i] > teeth[i] or  # price above teeth
                close[i] > ema_34_aligned[i]):  # 1d uptrend
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals