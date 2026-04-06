#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Elder Ray with 1-day trend filter and volume confirmation
# Elder Ray calculates Bull Power (High - EMA) and Bear Power (Low - EMA)
# Long when Bull Power > 0 and rising, Bear Power < 0 and rising (bullish divergence)
# Short when Bear Power < 0 and falling, Bull Power > 0 and falling (bearish divergence)
# Uses 1-day EMA13 for trend filter to ensure alignment with higher timeframe trend
# Volume confirmation requires volume > 1.3x 20-period average
# Designed for low trade frequency (target: 50-150 total trades over 4 years)
# Works in bull markets via Bull Power strength and bear markets via Bear Power divergence

name = "6h_elder_ray_1d_ema13_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for EMA13 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # EMA13 calculation on 1d
    ema13_1d = pd.Series(close_1d).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Align 1d EMA13 to 6h timeframe
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    
    # EMA13 for Elder Ray calculation (6h timeframe)
    ema13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # Bull Power: High - EMA
    bear_power = low - ema13   # Bear Power: Low - EMA
    
    # Slope of Bull Power and Bear Power (1-period change)
    bull_power_slope = np.diff(bull_power, prepend=bull_power[0])
    bear_power_slope = np.diff(bear_power, prepend=bear_power[0])
    
    # Volume average (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(13, n):
        # Skip if required data not available
        if (np.isnan(ema13_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(bull_power_slope[i]) or 
            np.isnan(bear_power_slope[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Bear Power turns positive (bullish momentum fading) or trend turns down
            elif bear_power[i] > 0 or close[i] < ema13_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Bull Power turns negative (bearish momentum fading) or trend turns up
            elif bull_power[i] < 0 or close[i] > ema13_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with Elder Ray signals and trend alignment
            # Long: Bull Power > 0 AND rising, Bear Power < 0 AND rising (bullish divergence)
            #        with 1-day uptrend and volume confirmation
            if (bull_power[i] > 0 and bull_power_slope[i] > 0 and
                bear_power[i] < 0 and bear_power_slope[i] > 0 and
                close[i] > ema13_1d_aligned[i] and
                volume[i] > 1.3 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: Bear Power < 0 AND falling, Bull Power > 0 AND falling (bearish divergence)
            #        with 1-day downtrend and volume confirmation
            elif (bear_power[i] < 0 and bear_power_slope[i] < 0 and
                  bull_power[i] > 0 and bull_power_slope[i] < 0 and
                  close[i] < ema13_1d_aligned[i] and
                  volume[i] > 1.3 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals