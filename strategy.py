#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 12h EMA50 trend filter and volume spike confirmation
# Long when Bull Power > 0 AND price > 12h EMA50 AND volume > 2.0x 24-bar avg
# Short when Bear Power < 0 AND price < 12h EMA50 AND volume > 2.0x 24-bar avg
# Exit when Elder Ray power crosses zero or opposite signal occurs
# Uses discrete position sizing (0.25) to minimize fee drag. Target: 15-25 trades/year on 6h.
# Elder Ray measures bull/bear strength relative to EMA13, working in both trending and ranging markets.
# 12h EMA50 filter ensures we only trade with the intermediate-term trend, improving win rate.
# Volume confirmation ensures signals have conviction, reducing false signals in choppy markets.

name = "6h_ElderRay_12hEMA50_Trend_VolumeSpike_v2"
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
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # Calculate EMA(50) on 12h data
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align EMA50 to 6h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate EMA(13) for Elder Ray (using 1d data for stability)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate EMA(13) on 1d data
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    # Align EMA13 to 6h timeframe
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # Calculate Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13_1d_aligned
    bear_power = low - ema_13_1d_aligned
    
    # Volume confirmation: >2.0x 24-bar average volume (4 periods on 6h = 1 day)
    volume_series = pd.Series(volume)
    volume_ma_24 = volume_series.rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > 2.0 * volume_ma_24
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 24  # Volume MA needs 24 bars
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(volume_ma_24[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_close = close[i]
        ema_50 = ema_50_12h_aligned[i]
        bull = bull_power[i]
        bear = bear_power[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when Bull Power > 0 AND price > 12h EMA50 AND volume confirmation
            if bull > 0 and curr_close > ema_50 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when Bear Power < 0 AND price < 12h EMA50 AND volume confirmation
            elif bear < 0 and curr_close < ema_50 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when Bull Power <= 0 or Bear Power >= 0 (zero crossover)
            if bull <= 0 or bear >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when Bull Power >= 0 or Bear Power <= 0 (zero crossover)
            if bull >= 0 or bear <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals