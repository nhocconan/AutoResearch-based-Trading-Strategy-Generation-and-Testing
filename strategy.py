#!/usr/bin/env python3
"""
1h_PositionBased_4hTrend
Hypothesis: On 1-hour timeframe, enter long when 4h EMA trend is bullish (close > EMA50) and price is near 1h VWAP support (within 0.3%), with volume confirmation. Enter short when 4h EMA trend is bearish (close < EMA50) and price is near 1h VWAP resistance (within 0.3%), with volume confirmation. Uses 4h trend for direction, 1h VWAP for precise entry, and volume filter to avoid false breakouts. Designed for low trade frequency (~20-40/year) to minimize fee decay while capturing trend continuations in both bull and bear markets.
"""

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
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    ema_50 = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    
    # 1h VWAP calculation (typical price * volume) / cumulative volume
    typical_price = (high + low + close) / 3.0
    vwap_numerator = np.cumsum(typical_price * volume)
    vwap_denominator = np.cumsum(volume)
    vwap = np.where(vwap_denominator > 0, vwap_numerator / vwap_denominator, close)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_aligned[i]) or np.isnan(vwap[i]) or 
            np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Calculate distance from VWAP as percentage
        vwap_distance_pct = abs(close[i] - vwap[i]) / vwap[i] * 100
        near_vwap = vwap_distance_pct <= 0.3  # Within 0.3% of VWAP
        
        # Trend conditions
        bullish_trend = close[i] > ema_50_aligned[i]
        bearish_trend = close[i] < ema_50_aligned[i]
        
        # Entry conditions with 4h trend alignment and volume surge
        long_entry = bullish_trend and near_vwap and volume_surge[i]
        short_entry = bearish_trend and near_vwap and volume_surge[i]
        
        # Exit conditions: trend reversal or VWAP breach
        long_exit = not bullish_trend or close[i] < vwap[i] * 0.997  # Below VWAP by 0.3%
        short_exit = not bearish_trend or close[i] > vwap[i] * 1.003  # Above VWAP by 0.3%
        
        if long_entry and position <= 0:
            signals[i] = 0.20
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.20
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.20  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.20   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_PositionBased_4hTrend"
timeframe = "1h"
leverage = 1.0