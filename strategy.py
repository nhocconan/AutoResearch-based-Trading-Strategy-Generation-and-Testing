#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + Elder Ray with 1w trend filter and volume confirmation
# Long when Alligator jaws < teeth < lips (bullish alignment) AND Elder Bull Power > 0 AND 1w close > 1w EMA50 AND volume > 2x 20-period average
# Short when Alligator jaws > teeth > lips (bearish alignment) AND Elder Bear Power < 0 AND 1w close < 1w EMA50 AND volume > 2x 20-period average
# Exit when Alligator alignment breaks (jaws-teeth-lips not in proper order) OR Elder power reverses sign
# Uses 12h primary timeframe with 1w HTF for trend filter
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 50-150 total trades over 4 years (12-37/year) based on proven slower TF performance
# Williams Alligator identifies trend alignment; Elder Ray measures bull/bear power; volume confirms momentum; 1w EMA50 filters for higher-timeframe trend
# Works in both bull and bear markets by following the 1w trend while using 12h for entry timing and Elder Ray for momentum confirmation

name = "12h_WilliamsAlligator_ElderRay_1wEMA50_Trend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 1w close for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Williams Alligator on 12h data
    # Jaw (blue line): 13-period SMMA, shifted 8 bars ahead
    # Teeth (red line): 8-period SMMA, shifted 5 bars ahead
    # Lips (green line): 5-period SMMA, shifted 3 bars ahead
    def smma(values, period):
        """Smoothed Moving Average"""
        if len(values) < period:
            return np.full_like(values, np.nan)
        result = np.full_like(values, np.nan)
        # First value is SMA
        result[period-1] = np.mean(values[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (PERIOD-1) + CLOSE) / PERIOD
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Shift as per Alligator specification
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    
    # Elder Ray on 12h data
    # Bull Power = High - EMA13
    # Bear Power = Low - EMA13
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volume confirmation: volume > 2x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Check Alligator alignment
        bullish_alignment = jaw[i] < teeth[i] < lips[i]
        bearish_alignment = jaw[i] > teeth[i] > lips[i]
        
        if position == 0:
            # Long conditions: bullish Alligator alignment AND Elder Bull Power > 0 AND 1w close > 1w EMA50 AND volume spike
            if (bullish_alignment and 
                bull_power[i] > 0 and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: bearish Alligator alignment AND Elder Bear Power < 0 AND 1w close < 1w EMA50 AND volume spike
            elif (bearish_alignment and 
                  bear_power[i] < 0 and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator alignment breaks OR Elder Bull Power <= 0
            if not (bullish_alignment and bull_power[i] > 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator alignment breaks OR Elder Bear Power >= 0
            if not (bearish_alignment and bear_power[i] < 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals