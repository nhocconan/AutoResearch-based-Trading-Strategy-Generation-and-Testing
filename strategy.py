#!/usr/bin/env python3
"""
4h_1d_Volume_Pullback_Strategy
Hypothesis: Buy pullbacks to 1-day VWAP in strong uptrends, sell rallies to VWAP in strong downtrends.
Trades with institutional flow using volume-weighted average price as dynamic support/resistance.
Works in bull markets (buy dips to VWAP in uptrends) and bear markets (sell rallies to VWAP in downtrends).
Uses 4-hour trend alignment with 1-day VWAP for institutional-grade entries. Target: 20-30 trades/year.
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
    
    # Get 1d data for VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate 1-day VWAP: typical price * volume / cumulative volume
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    vwap_1d = (typical_price * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    vwap_1d = vwap_1d.replace(0, np.nan).ffill().bfill().fillna(typical_price.iloc[0]).values
    
    # Align 1-day VWAP to 4h timeframe
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # 4h trend: 20-period EMA
    close_s = pd.Series(close)
    ema_20 = close_s.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_filter = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(100, n):
        # Skip if any required data is not ready
        if (np.isnan(vwap_1d_aligned[i]) or np.isnan(ema_20[i]) or 
            np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        # Long: price pulls back to VWAP in uptrend (price above EMA20)
        long_condition = (close[i] >= vwap_1d_aligned[i] * 0.998) and \
                         (close[i] <= vwap_1d_aligned[i] * 1.002) and \
                         (close[i] > ema_20[i]) and volume_filter[i]
        
        # Short: price rallies to VWAP in downtrend (price below EMA20)
        short_condition = (close[i] >= vwap_1d_aligned[i] * 0.998) and \
                          (close[i] <= vwap_1d_aligned[i] * 1.002) and \
                          (close[i] < ema_20[i]) and volume_filter[i]
        
        if long_condition and position != 1:
            position = 1
            signals[i] = position_size
        elif short_condition and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "4h_1d_Volume_Pullback_Strategy"
timeframe = "4h"
leverage = 1.0