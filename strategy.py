#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray combination with 1d EMA34 trend filter
# Williams Alligator (jaw/teeth/lips) identifies trend direction and strength
# Elder Ray (Bull/Bear Power) measures bull/bear power relative to EMA13
# 1d EMA34 ensures alignment with daily trend to avoid counter-trend trades
# Volume confirmation (>1.5x 20-period EMA) filters for institutional participation
# Designed for 6h timeframe targeting 12-37 trades/year (50-150 total over 4 years)
# Works in bull markets (alligator aligned up + bull power > 0) and bear markets (alligator aligned down + bear power < 0)
# Uses discrete position sizing (0.25) to balance return potential with drawdown control

name = "6h_WilliamsAlligator_ElderRay_1dEMA34_Trend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator (6h timeframe)
    # Jaw: Blue line - SMMA(13, 8)
    # Teeth: Red line - SMMA(8, 5)
    # Lips: Green line - SMMA(5, 3)
    def smma(arr, period):
        """Smoothed Moving Average"""
        result = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < period:
            return result
        # First value is simple SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CLOSE) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)  # SMMA(13, 8) - but we use period 13 for SMMA calculation
    teeth = smma(close, 8)  # SMMA(8, 5)
    lips = smma(close, 5)   # SMMA(5, 3)
    
    # Elder Ray (6h timeframe)
    # Bull Power = High - EMA13
    # Bear Power = Low - EMA13
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volume confirmation
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmation = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 1d EMA34
        bullish_bias = close[i] > ema_34_1d_aligned[i]
        bearish_bias = close[i] < ema_34_1d_aligned[i]
        
        # Alligator alignment: Lips > Teeth > Jaw = bullish, Lips < Teeth < Jaw = bearish
        bullish_alligator = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_alligator = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        if position == 0:  # Flat - look for new entries
            if bullish_bias and bullish_alligator and bull_power[i] > 0 and volume_confirmation[i]:
                # Long: Daily trend up, alligator aligned up, bull power positive, volume confirmation
                signals[i] = 0.25
                position = 1
            elif bearish_bias and bearish_alligator and bear_power[i] < 0 and volume_confirmation[i]:
                # Short: Daily trend down, alligator aligned down, bear power negative, volume confirmation
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Daily trend turns bearish OR alligator loses alignment OR bull power turns negative
            if (not bullish_bias) or (not bullish_alligator) or (bull_power[i] <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Daily trend turns bullish OR alligator loses alignment OR bear power turns positive
            if (not bearish_bias) or (not bearish_alligator) or (bear_power[i] >= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals