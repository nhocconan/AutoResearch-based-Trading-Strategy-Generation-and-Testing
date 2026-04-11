#!/usr/bin/env python3
"""
6h_12h_1d_cci_extreme_v2
Strategy: 6h CCI extreme + 12h/1d trend filter + volume confirmation
Timeframe: 6h
Leverage: 1.0
Hypothesis: Buy when 6h CCI < -150 (oversold) with 12h uptrend and volume > 1.5x average; sell when CCI > 150 (overbought) with 12h downtrend and volume confirmation. Uses 1d close > prior 1d close for uptrend definition. Designed to capture mean reversion in ranging markets while avoiding counter-trend trades. Low-frequency design targets 20-50 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_1d_cci_extreme_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_12h) < 20 or len(df_1d) < 20:
        return np.zeros(n)
    
    # 6h CCI (20-period)
    tp = (high + low + close) / 3.0
    ma = pd.Series(tp).rolling(window=20, min_periods=20).mean().values
    mad = pd.Series(np.abs(tp - ma)).rolling(window=20, min_periods=20).mean().values
    cci = (tp - ma) / (0.015 * mad)
    
    # 6h volume filter: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === 1d Close trend filter: use prior 1d's close ===
    close_1d = df_1d['close'].values
    close_1d_shifted = np.roll(close_1d, 1)
    close_1d_shifted[0] = np.nan
    close_1d_trend = align_htf_to_ltf(prices, df_1d, close_1d_shifted)
    
    # === 12h trend filter: use prior 12h close ===
    close_12h = df_12h['close'].values
    close_12h_shifted = np.roll(close_12h, 1)
    close_12h_shifted[0] = np.nan
    close_12h_trend = align_htf_to_ltf(prices, df_12h, close_12h_shifted)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(cci[i]) or np.isnan(vol_ma_20[i]) or
            np.isnan(close_1d_trend[i]) or np.isnan(close_12h_trend[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation: 6h volume must be elevated
        volume_confirmed = volume_current > 1.5 * vol_ma
        
        # Trend filters
        uptrend_1d = price_close > close_1d_trend[i]
        downtrend_1d = price_close < close_1d_trend[i]
        uptrend_12h = price_close > close_12h_trend[i]
        downtrend_12h = price_close < close_12h_trend[i]
        
        # Long conditions: CCI < -150 (oversold) + volume + 1d/12h uptrend
        long_signal = (cci[i] < -150) and volume_confirmed and uptrend_1d and uptrend_12h
        
        # Short conditions: CCI > 150 (overbought) + volume + 1d/12h downtrend
        short_signal = (cci[i] > 150) and volume_confirmed and downtrend_1d and downtrend_12h
        
        # Exit when CCI returns to neutral zone (-50 to 50)
        exit_long = position == 1 and cci[i] > -50
        exit_short = position == -1 and cci[i] < 50
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Buy when 6h CCI < -150 (oversold) with 12h uptrend and volume > 1.5x average; sell when CCI > 150 (overbought) with 12h downtrend and volume confirmation. Uses 1d close > prior 1d close for uptrend definition. Designed to capture mean reversion in ranging markets while avoiding counter-trend trades. Low-frequency design targets 20-50 trades/year to minimize fee drag.