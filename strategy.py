#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index with 12h regime filter
# Elder Ray = Bull Power (High - EMA13) and Bear Power (Low - EMA13)
# Long when Bull Power > 0 and Bear Power > -threshold (weak bearish pressure)
# Short when Bear Power < 0 and Bull Power < threshold (weak bullish pressure)
# 12h trend filter: only trade in direction of 12h EMA20 trend
# Volume confirmation: volume > 20-period average
# Target: 50-150 total trades over 4 years with controlled risk in both bull and bear markets
# Uses 6h timeframe with 12h trend filter to reduce false signals

name = "6h_elderray_12h_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_12h_slope = pd.Series(ema_12h).diff().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    ema_12h_slope_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_slope)
    
    # Elder Ray components (13-period EMA)
    ema13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(ema_12h_slope_aligned[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR approximation using price range
            if close[i] < entry_price - 2.0 * (high[i] - low[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Bear power becomes strongly negative or 12h trend turns down
            elif bear_power[i] < -0.5 * np.std(bear_power[max(0, i-20):i+1]) or ema_12h_slope_aligned[i] < 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2 * ATR approximation
            if close[i] > entry_price + 2.0 * (high[i] - low[i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Bull power becomes strongly positive or 12h trend turns up
            elif bull_power[i] > 0.5 * np.std(bull_power[max(0, i-20):i+1]) or ema_12h_slope_aligned[i] > 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and 12h trend alignment
            if vol_filter[i]:
                # Long when bull power positive, bear power not too negative, and 12h trend up
                if (bull_power[i] > 0 and 
                    bear_power[i] > -0.3 * np.std(bull_power[max(0, i-20):i+1]) and
                    ema_12h_slope_aligned[i] > 0):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short when bear power negative, bull power not too positive, and 12h trend down
                elif (bear_power[i] < 0 and 
                      bull_power[i] < 0.3 * np.std(bear_power[max(0, i-20):i+1]) and
                      ema_12h_slope_aligned[i] < 0):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals