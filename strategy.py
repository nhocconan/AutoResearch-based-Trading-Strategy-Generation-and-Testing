#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator + 1w EMA trend filter for multi-timeframe alignment
# Williams Alligator (13,8,5 SMAs with shifts) identifies trendless markets when jaws/lips/teeth are intertwined
# 1w EMA34 provides higher timeframe trend direction (long when price > EMA34, short when price < EMA34)
# Volume confirmation: current volume > 2.0x 50-period MA to ensure strong participation
# Entry: Long when Alligator is bullish (lips>teeth>jaws) AND price > 1w EMA34 AND volume spike
# Entry: Short when Alligator is bearish (lips<teeth<jaws) AND price < 1w EMA34 AND volume spike
# Exit: When Alligator becomes neutral (jaws between lips and teeth) OR price crosses 1w EMA34 in opposite direction
# Uses Alligator for trend strength and direction on 1d, 1w EMA for HTF filter, volume for conviction
# Timeframe: 1d, HTF: 1w. Target: 30-100 total trades over 4 years (7-25/year) to avoid fee drag.

name = "1d_WilliamsAlligator_1wEMA34_VolumeConfirm"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for EMA34
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # need sufficient data for EMA
        return np.zeros(n)
    
    # Calculate 1w EMA34
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Williams Alligator on 1d
    # Jaw: 13-period SMMA shifted 8 bars
    # Teeth: 8-period SMMA shifted 5 bars  
    # Lips: 5-period SMMA shifted 3 bars
    def smma(source, period):
        """Smoothed Moving Average"""
        if len(source) < period:
            return np.full(len(source), np.nan)
        result = np.full(len(source), np.nan)
        sma = pd.Series(source).rolling(window=period, min_periods=period).mean().values
        result[period-1] = sma[period-1]
        for i in range(period, len(source)):
            if np.isnan(result[i-1]) or np.isnan(source[i]):
                result[i] = np.nan
            else:
                result[i] = (result[i-1] * (period-1) + source[i]) / period
        return result
    
    jaw = smma((high + low) / 2, 13)  # Median price
    jaw_shifted = np.roll(jaw, 8)
    jaw_shifted[:8] = np.nan
    
    teeth = smma((high + low) / 2, 8)
    teeth_shifted = np.roll(teeth, 5)
    teeth_shifted[:5] = np.nan
    
    lips = smma((high + low) / 2, 5)
    lips_shifted = np.roll(lips, 3)
    lips_shifted[:3] = np.nan
    
    # Volume confirmation on 1d
    if len(volume) >= 50:
        vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
        volume_spike = volume > (2.0 * vol_ma_50)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or 
            np.isnan(lips_shifted[i]) or np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator conditions
        # Bullish: lips > teeth > jaw
        # Bearish: lips < teeth < jaw
        # Neutral: otherwise (jaws between lips and teeth or intertwined)
        bullish = lips_shifted[i] > teeth_shifted[i] and teeth_shifted[i] > jaw_shifted[i]
        bearish = lips_shifted[i] < teeth_shifted[i] and teeth_shifted[i] < jaw_shifted[i]
        
        if position == 0:
            # Long conditions: Bullish Alligator AND price > 1w EMA34 AND volume spike
            if bullish and close[i] > ema_34_1w_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short conditions: Bearish Alligator AND price < 1w EMA34 AND volume spike
            elif bearish and close[i] < ema_34_1w_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator turns neutral OR price crosses below 1w EMA34
            if not bullish or close[i] <= ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator turns neutral OR price crosses above 1w EMA34
            if not bearish or close[i] >= ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals