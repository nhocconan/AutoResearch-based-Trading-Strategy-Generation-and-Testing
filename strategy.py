#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1-day Keltner Channel breakout with volume confirmation
# Long when price closes above upper Keltner band (EMA20 + 2*ATR) with volume > 1.5x average
# Short when price closes below lower Keltner band (EMA20 - 2*ATR) with volume > 1.5x average
# Uses daily Keltner channels for dynamic support/resistance, volume for confirmation
# Designed to capture breakouts in trending markets while avoiding false signals in ranging conditions
# Target: 15-25 trades per year (60-100 over 4 years) with 0.25 position sizing

name = "12h_1dKeltner2x_Volume_Confirmation"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1-day Keltner Channel components
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # EMA20 of close
    close_series = pd.Series(df_1d['close'])
    ema20 = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # ATR(10) for Keltner width
    high_low = df_1d['high'] - df_1d['low']
    high_close = np.abs(df_1d['high'] - np.concatenate([[df_1d['close'][0]], df_1d['close'][:-1]]))
    low_close = np.abs(df_1d['low'] - np.concatenate([[df_1d['close'][0]], df_1d['close'][:-1]]))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Keltner bands: EMA20 ± 2*ATR10
    upper_keltner = ema20 + 2 * atr10
    lower_keltner = ema20 - 2 * atr10
    
    # Align Keltner levels to 12h timeframe
    upper_keltner_aligned = align_htf_to_ltf(prices, df_1d, upper_keltner)
    lower_keltner_aligned = align_htf_to_ltf(prices, df_1d, lower_keltner)
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after EMA/ATR warmup
        # Skip if any critical value is NaN or outside session
        if (np.isnan(upper_keltner_aligned[i]) or np.isnan(lower_keltner_aligned[i]) or 
            np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price closes above upper Keltner with volume confirmation
            if close[i] > upper_keltner_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short breakout: price closes below lower Keltner with volume confirmation
            elif close[i] < lower_keltner_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below lower Keltner (support break)
            if close[i] < lower_keltner_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above upper Keltner (resistance break)
            if close[i] > upper_keltner_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals