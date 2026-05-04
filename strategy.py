#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d ADX Regime + Volume Spike
# Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures trend strength.
# Long when Bull Power > 0 and rising AND Bear Power < 0 (bullish market structure).
# Short when Bear Power > 0 and rising AND Bull Power < 0 (bearish market structure).
# 1d ADX > 25 confirms trending regime to avoid false signals in ranging markets.
# Volume spike (2x 20-period EMA) ensures participation.
# Designed for 12-37 trades/year on 6h to minimize fee drag while capturing strong trends.
# Works in bull markets via long signals in uptrend and bear markets via short signals in downtrend.

name = "6h_ElderRay_1dADX_Regime_VolumeSpike"
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
    
    # Get 1d data for HTF ADX trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 25:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX for trend filter
    # True Range
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - pd.Series(close_1d).shift(1)))
    tr3 = pd.Series(np.abs(low_1d - pd.Series(close_1d).shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    
    # Directional Movement
    up_move = pd.Series(high_1d).diff()
    down_move = pd.Series(low_1d).diff()
    up_move = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    down_move = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    
    # Directional Indicators
    plus_di = 100 * (up_move.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr)
    minus_di = 100 * (down_move.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr)
    
    # ADX
    dx = (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    adx_values = adx.values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    
    # Calculate Elder Ray on 6h data
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean()
    bull_power = high - ema_13.values  # High - EMA13
    bear_power = ema_13.values - low   # EMA13 - Low
    
    # Slope of Bull/Bear Power (3-period EMA of the power)
    bull_power_slope = pd.Series(bull_power).ewm(span=3, adjust=False, min_periods=3).mean().diff()
    bear_power_slope = pd.Series(bear_power).ewm(span=3, adjust=False, min_periods=3).mean().diff()
    
    bull_power_values = bull_power
    bear_power_values = bear_power
    bull_power_slope_values = bull_power_slope.fillna(0).values
    bear_power_slope_values = bear_power_slope.fillna(0).values
    
    # Calculate volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 2.0)  # Volume at least 2x average for confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(bull_power_values[i]) or 
            np.isnan(bear_power_values[i]) or np.isnan(bull_power_slope_values[i]) or 
            np.isnan(bear_power_slope_values[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Bull Power > 0 and rising AND Bear Power < 0 AND ADX > 25 AND volume spike
            if (bull_power_values[i] > 0 and 
                bull_power_slope_values[i] > 0 and 
                bear_power_values[i] < 0 and 
                adx_aligned[i] > 25 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Bear Power > 0 and rising AND Bull Power < 0 AND ADX > 25 AND volume spike
            elif (bear_power_values[i] > 0 and 
                  bear_power_slope_values[i] > 0 and 
                  bull_power_values[i] < 0 and 
                  adx_aligned[i] > 25 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power <= 0 OR Bear Power >= 0 OR ADX < 20 (regime change)
            if (bull_power_values[i] <= 0 or 
                bear_power_values[i] >= 0 or 
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power <= 0 OR Bull Power >= 0 OR ADX < 20 (regime change)
            if (bear_power_values[i] <= 0 or 
                bull_power_values[i] >= 0 or 
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals