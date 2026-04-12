#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h_1d_camarilla_breakout_v32 - Enhanced version with tighter entry conditions and improved risk management
# Uses Camarilla levels from daily chart with volume confirmation, volatility filter, and trend filter.
# Designed to work in both bull and bear markets by capturing breakouts/breakdowns with institutional volume.
# Target: 20-40 trades/year per symbol for low friction and high win rate.

name = "4h_1d_camarilla_breakout_v32"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    # Camarilla formulas
    range_prev = high_prev - low_prev
    camarilla_h4 = close_prev + range_prev * 1.1 / 2
    camarilla_l4 = close_prev - range_prev * 1.1 / 2
    
    # Align to 4h timeframe (already delayed by 1 day due to shift)
    h4_level = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_level = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Volume confirmation: volume > 1.8 * 20-period average (stricter than before)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.8)
    
    # Trend filter: price > 50-period EMA for longs, price < 50-period EMA for shorts
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_filter_long = close > ema_50
    trend_filter_short = close < ema_50
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup
        # Skip if levels not ready
        if np.isnan(h4_level[i]) or np.isnan(l4_level[i]):
            signals[i] = 0.0
            continue
        
        # Check volume and trend filters
        vol_ok = vol_confirm[i]
        long_trend_ok = trend_filter_long[i]
        short_trend_ok = trend_filter_short[i]
        
        # Long signal: price breaks above H4 with volume and uptrend
        if vol_ok and long_trend_ok and close[i] > h4_level[i] and position != 1:
            position = 1
            signals[i] = 0.30
        # Short signal: price breaks below L4 with volume and downtrend
        elif vol_ok and short_trend_ok and close[i] < l4_level[i] and position != -1:
            position = -1
            signals[i] = -0.30
        # Exit conditions: opposite breakout or trend reversal
        elif position == 1 and (close[i] < l4_level[i] or close[i] < ema_50[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > h4_level[i] or close[i] > ema_50[i]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.30
            elif position == -1:
                signals[i] = -0.30
            else:
                signals[i] = 0.0
    
    return signals