#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_camarilla_pivot_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 4h trend: EMA(50) slope
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    ema_slope = np.zeros(len(ema_50_4h_aligned))
    ema_slope[1:] = ema_50_4h_aligned[1:] - ema_50_4h_aligned[:-1]
    
    # Calculate 1d Camarilla Pivot
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    range_1d = high_1d - low_1d
    camarilla_h4 = close_1d + (range_1d * 1.1 / 2)
    camarilla_l4 = close_1d - (range_1d * 1.1 / 2)
    camarilla_h3 = close_1d + (range_1d * 1.1 / 4)
    camarilla_l3 = close_1d - (range_1d * 1.1 / 4)
    camarilla_h2 = close_1d + (range_1d * 1.1 / 6)
    camarilla_l2 = close_1d - (range_1d * 1.1 / 6)
    
    # Align 1d Camarilla levels to 1h
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    h2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h2)
    l2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l2)
    
    # Volume filter: volume > 1.2x 24-period average
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if any required data is invalid
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(h2_aligned[i]) or np.isnan(l2_aligned[i]) or
            np.isnan(ema_slope[i]) or np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        # Session filter
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        vol_ma = vol_ma_24[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.2 * vol_ma
        
        # Trend filter: 4h EMA slope positive for long, negative for short
        trend_up = ema_slope[i] > 0
        trend_down = ema_slope[i] < 0
        
        # Long conditions: price crosses above H3 with volume and trend up
        long_signal = (price_close > h3_aligned[i] and price_low <= h3_aligned[i] and 
                      volume_confirmed and trend_up)
        
        # Short conditions: price crosses below L3 with volume and trend down
        short_signal = (price_close < l3_aligned[i] and price_high >= l3_aligned[i] and 
                       volume_confirmed and trend_down)
        
        # Exit conditions: price returns to H2/L2 or opposite H3/L3
        exit_long = (position == 1 and 
                    (price_close <= h2_aligned[i] or price_close >= h3_aligned[i]))
        exit_short = (position == -1 and 
                     (price_close >= l2_aligned[i] or price_close <= l3_aligned[i]))
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Camarilla pivot breakout strategy on 1h timeframe with 4h trend filter and 1d Camarilla levels.
# Uses 1d Camarilla H3/L3 levels for breakout entries, 4h EMA slope for trend filter, and volume confirmation.
# Enters long when price crosses above H3 with volume and uptrend, short when crosses below L3 with volume and downtrend.
# Exits when price returns to H2/L2 or breaks through H3/L3 in the opposite direction.
# Session filter (08-20 UTC) reduces noise trades. Designed for 15-30 trades/year to minimize fee drag.
# Works in both bull and bear markets by trading breakouts in the direction of the 4h trend.
# Camarilla levels provide natural support/resistance that work across market regimes.