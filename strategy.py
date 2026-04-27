#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Elder Ray with 1d trend filter and volume confirmation.
# Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13.
# Long when Bull Power > 0 and rising, Bear Power < 0, with volume > 1.5x average and 1d close > EMA50.
# Short when Bear Power < 0 and falling, Bull Power < 0, with volume > 1.5x average and 1d close < EMA50.
# Exit when power signals reverse or volume drops below average.
# Designed for ~20-30 trades/year with strong trend and volume filters to avoid whipsaws.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter and Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate EMA13 for Elder Ray on 1d data
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components on 1d data
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = low_1d - ema13_1d
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d indicators to 12h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: volume > 1.5x 20-period average (20*12h = 10 days)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need 20-period volume MA and EMA13
    start_idx = max(20, 13)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema50_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Elder Ray signals
        bull_power = bull_power_aligned[i]
        bear_power = bear_power_aligned[i]
        
        # Trend filter from 1d EMA50
        bullish_trend = price > ema50_aligned[i]
        bearish_trend = price < ema50_aligned[i]
        
        if position == 0:
            # Long: Bull Power positive and rising, Bear Power negative, with volume and bullish trend
            if bull_power > 0 and i > start_idx and bull_power > bull_power_aligned[i-1] and \
               bear_power < 0 and vol_filter and bullish_trend:
                signals[i] = size
                position = 1
            # Short: Bear Power negative and falling, Bull Power negative, with volume and bearish trend
            elif bear_power < 0 and i > start_idx and bear_power < bear_power_aligned[i-1] and \
                 bull_power < 0 and vol_filter and bearish_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Bull Power turns negative or Bear Power turns positive or volume drops
            if bull_power <= 0 or bear_power >= 0 or vol_now <= vol_avg:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: Bear Power turns positive or Bull Power turns negative or volume drops
            if bear_power >= 0 or bull_power <= 0 or vol_now <= vol_avg:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_ElderRay_Volume_1dTrend"
timeframe = "12h"
leverage = 1.0