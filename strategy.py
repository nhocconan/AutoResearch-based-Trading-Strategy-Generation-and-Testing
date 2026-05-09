#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + Daily Trend + Volume Spike
# Bull Power = High - EMA13; Bear Power = Low - EMA13
# Long when Bull Power > 0, Bear Power rising, price > daily EMA50, volume > 1.5x avg
# Short when Bear Power < 0, Bull Power falling, price < daily EMA50, volume > 1.5x avg
# Exit when power crosses zero or volume drops. Works in bull/bear via trend filter.
# Target: 12-37 trades/year (50-150 over 4 years) with strict volume + trend confluence.

name = "6h_ElderRay_Power_Trend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Elder Ray, trend, and volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily EMA13 for Elder Ray power calculation
    close_1d = df_1d['close'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = Daily High - EMA13
    bull_power = df_1d['high'].values - ema13_1d
    # Bear Power = Daily Low - EMA13
    bear_power = df_1d['low'].values - ema13_1d
    
    # Daily EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume filter: current daily volume > 1.5 * 20-day average
    vol_series = pd.Series(df_1d['volume'].values)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter_1d = df_1d['volume'].values > (vol_ma * 1.5)
    
    # Align all to 6h timeframe
    bull_power_6h = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_6h = align_htf_to_ltf(prices, df_1d, bear_power)
    ema50_1d_6h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    volume_filter_6h = align_htf_to_ltf(prices, df_1d, volume_filter_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Need enough data for EMA50 and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(bull_power_6h[i]) or np.isnan(bear_power_6h[i]) or
            np.isnan(ema50_1d_6h[i]) or np.isnan(volume_filter_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        bull_power_val = bull_power_6h[i]
        bear_power_val = bear_power_6h[i]
        trend = ema50_1d_6h[i]
        vol_filter = volume_filter_6h[i]
        
        if position == 0:
            # Enter long: Bull Power > 0 (bullish), Bear Power rising (less negative), 
            # price above daily trend, volume spike
            if (bull_power_val > 0 and 
                i > start_idx and bear_power_val > bear_power_6h[i-1] and
                close[i] > trend and vol_filter):
                signals[i] = 0.25
                position = 1
            # Enter short: Bear Power < 0 (bearish), Bull Power falling (less positive),
            # price below daily trend, volume spike
            elif (bear_power_val < 0 and 
                  i > start_idx and bull_power_val < bull_power_6h[i-1] and
                  close[i] < trend and vol_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Bull Power turns negative OR volume drops
            if bull_power_val <= 0 or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Bear Power turns positive OR volume drops
            if bear_power_val >= 0 or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals