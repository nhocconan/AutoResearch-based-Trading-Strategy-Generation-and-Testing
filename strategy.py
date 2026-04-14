#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Bollinger Band width expansion + 1-day trend + volume confirmation
# Uses Bollinger Band width (BBW) on 12h to detect volatility expansion
# Long when BBW increases and price > upper band; Short when BBW increases and price < lower band
# Daily close > daily EMA50 as trend filter (only long in daily uptrend, short in daily downtrend)
# Volume confirmation > 1.3x 20-period EMA on 12h to reduce false signals
# Designed for ~15-25 trades/year with clear volatility-based logic
# Works in bull markets via uptrend + expansion and in bear markets via downtrend + expansion

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Bollinger Bands (20-period, 2 std dev) for 12h
    close_series = pd.Series(close)
    bb_middle = close_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = bb_upper - bb_lower
    
    # Calculate BBW change (current - previous)
    bb_width_change = np.diff(bb_width, prepend=bb_width[0])
    
    # Load daily data once for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume moving average for confirmation (20-period EMA on 12h)
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(20, n):
        # Get aligned daily EMA50
        ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)[i]
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or np.isnan(bb_width_change[i]) or np.isnan(ema50_1d_aligned) or np.isnan(vol_ma[i]):
            continue
        
        # Daily trend filter: only long in uptrend, only short in downtrend
        price_vs_ema = close[i] > ema50_1d_aligned  # Simple price vs EMA comparison
        
        # Bollinger Band signals
        bb_expansion = bb_width_change[i] > 0  # BB width increasing
        price_above_upper = close[i] > bb_upper[i]
        price_below_lower = close[i] < bb_lower[i]
        
        # Volume confirmation (1.3x average)
        volume_confirm = volume[i] > 1.3 * vol_ma[i]
        
        # Long signal: BB expansion + price above upper band + daily uptrend + volume
        if position == 0 and bb_expansion and price_above_upper and price_vs_ema and volume_confirm:
            position = 1
            signals[i] = position_size
        # Short signal: BB expansion + price below lower band + daily downtrend + volume
        elif position == 0 and bb_expansion and price_below_lower and not price_vs_ema and volume_confirm:
            position = -1
            signals[i] = -position_size
        # Exit: BB contraction or price crosses middle band
        elif position != 0:
            if position == 1 and (not bb_expansion or close[i] < bb_middle[i]):
                position = 0
                signals[i] = 0.0
            elif position == -1 and (not bb_expansion or close[i] > bb_middle[i]):
                position = 0
                signals[i] = 0.0
    
    return signals

name = "12h_BollingerWidth_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0