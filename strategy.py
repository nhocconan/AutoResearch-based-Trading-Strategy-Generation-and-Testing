#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Camarilla pivot long/short with 1d trend filter + volume spike
    # Uses 1d EMA50 for trend filter: long only when price > EMA50, short only when price < EMA50
    # Entry at Camarilla H3/L3 levels with volume > 2.0 * 20-period average
    # Exit at Camarilla H4/L4 or opposite H3/L3 touch
    # Discrete sizing 0.25 to minimize fee churn. Target: 20-40 trades/year per symbol.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate previous day's Camarilla levels (H3, L3, H4, L4)
    # H4 = close + 1.5*(high-low), L4 = close - 1.5*(high-low)
    # H3 = close + 1.125*(high-low), L3 = close - 1.125*(high-low)
    camarilla_high = np.full(len(df_1d), np.nan)
    camarilla_low = np.full(len(df_1d), np.nan)
    camarilla_h4 = np.full(len(df_1d), np.nan)
    camarilla_l4 = np.full(len(df_1d), np.nan)
    
    for i in range(1, len(df_1d)):
        high_val = high_1d[i-1]
        low_val = low_1d[i-1]
        close_val = close_1d[i-1]
        rang = high_val - low_val
        camarilla_high[i] = close_val + 1.125 * rang  # H3
        camarilla_low[i] = close_val - 1.125 * rang   # L3
        camarilla_h4[i] = close_val + 1.5 * rang      # H4
        camarilla_l4[i] = close_val - 1.5 * rang      # L4
    
    # Align Camarilla levels to 4h timeframe
    camarilla_high_aligned = align_htf_to_ltf(prices, df_1d, camarilla_high)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_1d, camarilla_low)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Volume confirmation: volume > 2.0 * 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(camarilla_high_aligned[i]) or 
            np.isnan(camarilla_low_aligned[i]) or np.isnan(camarilla_h4_aligned[i]) or 
            np.isnan(camarilla_l4_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine 1d trend
        bullish_trend = close[i] > ema50_1d_aligned[i]
        bearish_trend = close[i] < ema50_1d_aligned[i]
        
        # Entry logic: Camarilla H3/L3 breakout with volume and trend filter
        long_entry = False
        short_entry = False
        
        # Long entry: price breaks above Camarilla H3 in bullish trend with volume spike
        if bullish_trend:
            long_entry = (close[i] > camarilla_high_aligned[i]) and volume_spike[i]
        # Short entry: price breaks below Camarilla L3 in bearish trend with volume spike
        elif bearish_trend:
            short_entry = (close[i] < camarilla_low_aligned[i]) and volume_spike[i]
        
        # Exit logic: Camarilla H4/L4 touch or opposite H3/L3 touch
        long_exit = (close[i] >= camarilla_h4_aligned[i]) or (close[i] <= camarilla_low_aligned[i])
        short_exit = (close[i] <= camarilla_l4_aligned[i]) or (close[i] >= camarilla_high_aligned[i])
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_camarilla_breakout_trend_volume_v1"
timeframe = "4h"
leverage = 1.0