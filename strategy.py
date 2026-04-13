#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Camarilla H3/L3 breakout with 1d EMA50 trend filter and volume spike confirmation.
    # Long when price breaks above H3 AND 1d close > EMA50 AND 4h volume > 2.0x 20-period MA.
    # Short when price breaks below L3 AND 1d close < EMA50 AND 4h volume > 2.0x 20-period MA.
    # Exit when price re-enters H3-L3 range.
    # Uses discrete position sizing (0.25) and strict volume filter (2.0x) to target 75-200 trades over 4 years.
    # Works in bull/bear via trend filter and volume confirmation reducing false breakouts.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels from previous day
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    h3_1d = pivot_1d + range_1d * 1.1 / 4
    l3_1d = pivot_1d - range_1d * 1.1 / 4
    h4_1d = pivot_1d + range_1d * 1.1 / 2
    l4_1d = pivot_1d - range_1d * 1.1 / 2
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d indicators to 4h timeframe
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Get 4h data for volume confirmation and price
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    volume_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    # Align 4h close for price comparison
    close_4h_aligned = align_htf_to_ltf(prices, df_4h, close)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_4h_aligned[i]) or 
            np.isnan(close_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 2.0x 20-period average (strict filter)
        volume_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_4h)
        volume_spike = volume_4h_aligned[i] > 2.0 * vol_ma_4h_aligned[i]
        
        # Price relative to Camarilla levels
        price_above_h3 = close_4h_aligned[i] > h3_1d_aligned[i]
        price_below_l3 = close_4h_aligned[i] < l3_1d_aligned[i]
        price_in_range = (close_4h_aligned[i] >= l3_1d_aligned[i]) & (close_4h_aligned[i] <= h3_1d_aligned[i])
        
        # Trend filter: 1d close vs EMA50 (using 4h close as proxy for 1d close trend)
        trend_bullish = close_4h_aligned[i] > ema50_1d_aligned[i]
        trend_bearish = close_4h_aligned[i] < ema50_1d_aligned[i]
        
        # Entry conditions
        if price_above_h3 and trend_bullish and volume_spike and position != 1:
            position = 1
            signals[i] = position_size
        elif price_below_l3 and trend_bearish and volume_spike and position != -1:
            position = -1
            signals[i] = -position_size
        # Exit conditions: price re-enters H3-L3 range
        elif price_in_range and position != 0:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_camarilla_breakout_ema_volume_spike_v2"
timeframe = "4h"
leverage = 1.0