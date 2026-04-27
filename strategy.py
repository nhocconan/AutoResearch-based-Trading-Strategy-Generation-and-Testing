#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Elder Ray Index (Bull/Bear Power) with 13-day EMA, filtered by 1d trend and volume spike.
# Bull Power = High - EMA13, Bear Power = Low - EMA13. 
# Long when Bull Power > 0 and rising, Bear Power < 0 and falling, with volume > 2x average and 1d close > EMA50.
# Short when Bear Power < 0 and falling, Bull Power < 0 and rising, with volume > 2x average and 1d close < EMA50.
# Exit when power crosses zero or trend reverses.
# Designed for ~20-30 trades/year with strong trend filtering to avoid whipsaws in ranging markets.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA13 for Elder Ray (13-period EMA of close)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: volume > 2x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need 20-period volume MA and 13-period EMA
    start_idx = max(20, 13)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema50_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter
        vol_filter = vol_now > 2.0 * vol_avg
        
        # Trend filters from 1d EMA50
        bullish_trend = price > ema50_aligned[i]
        bearish_trend = price < ema50_aligned[i]
        
        if position == 0:
            # Long: Bull Power > 0 and rising, Bear Power < 0 and falling, with volume and bullish trend
            if (bull_power[i] > 0 and bull_power[i] > bull_power[i-1] and
                bear_power[i] < 0 and bear_power[i] < bear_power[i-1] and
                vol_filter and bullish_trend):
                signals[i] = size
                position = 1
            # Short: Bear Power < 0 and falling, Bull Power < 0 and rising, with volume and bearish trend
            elif (bear_power[i] < 0 and bear_power[i] < bear_power[i-1] and
                  bull_power[i] < 0 and bull_power[i] > bull_power[i-1] and
                  vol_filter and bearish_trend):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Bull Power crosses below zero or trend turns bearish
            if bull_power[i] <= 0 or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: Bear Power crosses above zero or trend turns bullish
            if bear_power[i] >= 0 or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_ElderRay_BullBearPower_Volume_1dTrend"
timeframe = "4h"
leverage = 1.0