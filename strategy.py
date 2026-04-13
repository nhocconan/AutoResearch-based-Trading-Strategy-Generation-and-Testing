#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Camarilla pivot breakout with 1d trend filter and volume confirmation.
    # Long when price breaks above Camarilla H3 level AND 1d close > 1d EMA50 (bullish trend) AND 12h volume > 1.3x 20-period MA.
    # Short when price breaks below Camarilla L3 level AND 1d close < 1d EMA50 (bearish trend) AND 12h volume > 1.3x 20-period MA.
    # Exit when price re-enters Camarilla H3-L3 range (mean reversion).
    # Uses Camarilla pivots for structure, 1d EMA for trend, volume for confirmation.
    # Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
    
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
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # H4 = Pivot + Range * 1.1/2
    # H3 = Pivot + Range * 1.1/4
    # H2 = Pivot + Range * 1.1/6
    # H1 = Pivot + Range * 1.1/12
    # L1 = Pivot - Range * 1.1/12
    # L2 = Pivot - Range * 1.1/6
    # L3 = Pivot - Range * 1.1/4
    # L4 = Pivot - Range * 1.1/2
    
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    h3_1d = pivot_1d + range_1d * 1.1 / 4
    l3_1d = pivot_1d - range_1d * 1.1 / 4
    h4_1d = pivot_1d + range_1d * 1.1 / 2
    l4_1d = pivot_1d - range_1d * 1.1 / 2
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d indicators to 12h timeframe
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Get 12h data for volume confirmation and price
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    volume_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # Align 12h close for price comparison
    close_12h_aligned = align_htf_to_ltf(prices, df_12h, close)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_12h_aligned[i]) or 
            np.isnan(close_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.3x 20-period average
        volume_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_12h)
        volume_spike = volume_12h_aligned[i] > 1.3 * vol_ma_12h_aligned[i]
        
        # Price relative to Camarilla levels
        price_above_h3 = close_12h_aligned[i] > h3_1d_aligned[i]
        price_below_l3 = close_12h_aligned[i] < l3_1d_aligned[i]
        price_in_range = (close_12h_aligned[i] >= l3_1d_aligned[i]) & (close_12h_aligned[i] <= h3_1d_aligned[i])
        
        # Trend filter: 1d close vs EMA50
        trend_bullish = close_12h_aligned[i] > ema50_1d_aligned[i]  # Using 12h close vs 1d EMA50
        trend_bearish = close_12h_aligned[i] < ema50_1d_aligned[i]
        
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

name = "12h_1d_camarilla_pivot_breakout_ema_volume_v1"
timeframe = "12h"
leverage = 1.0