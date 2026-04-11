#!/usr/bin/env python3
"""
4h_1d_squeeze_breakout_volume_v1
Strategy: 4h Bollinger Band squeeze breakout with volume confirmation and 1d trend filter
Timeframe: 4h
Leverage: 1.0
Hypothesis: Uses Bollinger Band squeeze detection (BB width < 20-day percentile) to identify low volatility periods. Breakouts from squeeze are confirmed by volume spike (>1.5x average) and filtered by 1d EMA50 trend direction. Works in bull markets (breakouts with trend) and bear markets (breakouts against trend filtered out). Designed to capture explosive moves after consolidation with tight stops. Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_squeeze_breakout_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h Bollinger Bands (20, 2)
    bb_middle = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2.0 * bb_std
    bb_lower = bb_middle - 2.0 * bb_std
    bb_width = bb_upper - bb_lower
    
    # Bollinger Band squeeze: width < 20-period percentile (20%)
    bb_width_percentile = pd.Series(bb_width).rolling(window=20, min_periods=20).quantile(0.2).values
    squeeze = bb_width < bb_width_percentile
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(bb_middle[i]) or np.isnan(bb_std[i]) or 
            np.isnan(vol_avg[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Trend filter: price above/below 1d EMA50
        uptrend_1d = price_close > ema_50_1d_aligned[i]
        downtrend_1d = price_close < ema_50_1d_aligned[i]
        
        # Breakout conditions: price breaks above/below Bollinger Bands
        breakout_up = price_close > bb_upper[i-1]  # Use previous bar's upper band
        breakout_down = price_close < bb_lower[i-1]  # Use previous bar's lower band
        
        # Volume confirmation
        vol_confirmed = vol_spike[i]
        
        # Squeeze condition: only trade when coming out of squeeze
        squeeze_active = squeeze[i-1]  # Was in squeeze on previous bar
        
        # Long: upward breakout with volume from squeeze in uptrend
        long_signal = breakout_up and vol_confirmed and squeeze_active and uptrend_1d
        
        # Short: downward breakout with volume from squeeze in downtrend
        short_signal = breakout_down and vol_confirmed and squeeze_active and downtrend_1d
        
        # Exit when price returns to middle (BB middle) or opposite band
        exit_long = position == 1 and (price_close < bb_middle[i] or price_close > bb_upper[i])
        exit_short = position == -1 and (price_close > bb_middle[i] or price_close < bb_lower[i])
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals