#!/usr/bin/env python3
"""
Hypothesis: 6-hour Bollinger Band squeeze breakout with 12-hour trend filter and volume confirmation.
Squeeze indicates low volatility breakout setup. Use 12h EMA50 for trend direction to avoid counter-trend trades.
Volume > 1.5x 20-period average confirms breakout strength.
Target: 20-40 trades/year per symbol (80-160 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_bbands(close, length=20, std_dev=2.0):
    """Calculate Bollinger Bands"""
    if len(close) < length:
        return np.full_like(close, np.nan), np.full_like(close, np.nan), np.full_like(close, np.nan)
    
    ma = pd.Series(close).rolling(window=length, min_periods=length).mean().values
    std = pd.Series(close).rolling(window=length, min_periods=length).std().values
    upper = ma + (std_dev * std)
    lower = ma - (std_dev * std)
    return upper, ma, lower

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 6h Bollinger Bands (20, 2)
    bb_upper, bb_middle, bb_lower = calculate_bbands(close, 20, 2.0)
    
    # Calculate 6h volume MA (20)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need 12h EMA50 (50) + BBands (20) + Vol MA (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_now = close[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Bollinger Band squeeze detection: bandwidth < 5% of price
        bb_width = bb_upper[i] - bb_lower[i]
        squeeze_condition = bb_width < (0.05 * price_now)
        
        # Volume filter: volume > 1.5x 20-period average
        vol_filter = vol_now > 1.5 * vol_ma
        
        # Trend filter: price vs 12h EMA50
        uptrend = price_now > ema_50_12h_aligned[i]
        downtrend = price_now < ema_50_12h_aligned[i]
        
        if position == 0:
            # Look for squeeze breakout with volume and trend alignment
            if squeeze_condition and vol_filter:
                if uptrend and price_now > bb_upper[i]:
                    signals[i] = size
                    position = 1
                elif downtrend and price_now < bb_lower[i]:
                    signals[i] = -size
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes below middle band or squeeze breaks opposite direction
            if price_now < bb_middle[i] or (squeeze_condition and price_now < bb_lower[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price closes above middle band or squeeze breaks opposite direction
            if price_now > bb_middle[i] or (squeeze_condition and price_now > bb_upper[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_BollingerSqueeze_12hEMA50_Trend_Volume"
timeframe = "6h"
leverage = 1.0