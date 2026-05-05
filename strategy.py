#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull Power/Bear Power) + 1d EMA34 trend filter + volume confirmation
# Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
# Long when Bull Power > 0 AND Bear Power < 0 AND close > 1d EMA34 AND volume > 1.5x 20-period average
# Short when Bull Power < 0 AND Bear Power > 0 AND close < 1d EMA34 AND volume > 1.5x 20-period average
# Exit when Bull Power and Bear Power have same sign (both positive or both negative) OR close crosses 1d EMA34
# Uses 6h primary timeframe with 1d HTF for trend filter to capture multi-day moves with controlled frequency
# Discrete sizing (0.25) to limit fee drag and manage drawdown in both bull and bear markets
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag
# Elder Ray measures bull/bear strength relative to EMA; EMA34 filters for higher-timeframe trend; volume confirms participation

name = "6h_Elder_Ray_1dEMA34_Trend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Elder Ray components on 6h data
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = low - ema13   # Bear Power = Low - EMA13
    
    # Volume confirmation: volume > 1.5x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Elder Ray conditions
        bullish_pressure = bull_power[i] > 0 and bear_power[i] < 0  # Bull Power > 0 AND Bear Power < 0
        bearish_pressure = bull_power[i] < 0 and bear_power[i] > 0  # Bull Power < 0 AND Bear Power > 0
        
        if position == 0:
            # Long conditions: Bullish pressure AND close > 1d EMA34 AND volume spike
            if bullish_pressure and close[i] > ema_34_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short conditions: Bearish pressure AND close < 1d EMA34 AND volume spike
            elif bearish_pressure and close[i] < ema_34_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bullish and Bear Power same sign OR close < 1d EMA34 (trend flip)
            if (bull_power[i] > 0 and bear_power[i] > 0) or (bull_power[i] < 0 and bear_power[i] < 0) or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bullish and Bear Power same sign OR close > 1d EMA34 (trend flip)
            if (bull_power[i] > 0 and bear_power[i] > 0) or (bull_power[i] < 0 and bear_power[i] < 0) or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals