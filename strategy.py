#!/usr/bin/env python3
# Hypothesis: 6h Williams %R with 1d volume regime filter. Long when Williams %R < -80 (oversold) AND 1d volume > 1.5x 20-period EMA volume (strong participation) AND close > 6h EMA34 (uptrend bias). Short when Williams %R > -20 (overbought) AND 1d volume > 1.5x 20-period EMA volume AND close < 6h EMA34. Uses discrete position sizing (0.25) to minimize fee churn. Designed to capture mean reversals in high-volume trending environments, avoiding low-volume whipsaws. Target: 15-35 trades/year.

name = "6h_WilliamsR_VolumeRegime_EMA34_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate EMA34 on 6h for trend filter
    close_s = pd.Series(close)
    ema34_6h = close_s.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Get 1d data for Williams %R and volume filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Williams %R(14) on 1d
    # Highest high and lowest low over 14 periods
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    # Handle division by zero (when highest_high == lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate EMA20 of 1d volume for volume filter
    volume_s = pd.Series(volume_1d)
    ema20_volume = volume_s.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align HTF arrays to 6h timeframe (wait for completed 1d bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema20_volume_aligned = align_htf_to_ltf(prices, df_1d, ema20_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema20_volume_aligned[i]) or 
            np.isnan(ema34_6h[i]) or np.isnan(atr[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R < -80 (oversold) AND volume > 1.5x EMA20 volume AND close > EMA34
            if williams_r_aligned[i] < -80 and volume[i] > 1.5 * ema20_volume_aligned[i] and close[i] > ema34_6h[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R > -20 (overbought) AND volume > 1.5x EMA20 volume AND close < EMA34
            elif williams_r_aligned[i] > -20 and volume[i] > 1.5 * ema20_volume_aligned[i] and close[i] < ema34_6h[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R > -50 (momentum fading) OR close < EMA34 (trend break)
            if williams_r_aligned[i] > -50 or close[i] < ema34_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R < -50 (momentum fading) OR close > EMA34 (trend break)
            if williams_r_aligned[i] < -50 or close[i] > ema34_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals