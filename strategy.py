#!/usr/bin/env python3
name = "1h_4h_1d_Camarilla_Squeeze_Breakout"
timeframe = "1h"
leverage = 1.0

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
    
    # Get 4h and 1d data for direction and regime
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 20 or len(df_1d) < 20:
        return np.zeros(n)
    
    # 4h Close for trend filter
    close_4h = df_4h['close'].values
    # 4h EMA20 for trend
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    trend_4h = close_4h > ema20_4h
    
    # 1d Bollinger Bands for squeeze detection (20, 2.0)
    close_1d = df_1d['close'].values
    sma20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma20_1d + 2 * std20_1d
    lower_bb = sma20_1d - 2 * std20_1d
    bb_width = (upper_bb - lower_bb) / sma20_1d
    # Squeeze: BB width below 20-period percentile 30%
    bb_width_percentile = pd.Series(bb_width).rolling(window=20, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0, raw=False
    ).values
    squeeze = bb_width_percentile < 0.3
    
    # 1h Camarilla levels (based on prior 1h bar)
    # We'll calculate these in the loop using prior bar data
    
    # Align higher timeframe signals to 1h
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    squeeze_aligned = align_htf_to_ltf(prices, df_1d, squeeze)
    
    # Pre-calculate volume average for confirmation
    vol_ma20 = np.zeros(n)
    for i in range(n):
        if i < 20:
            vol_ma20[i] = np.mean(volume[:i+1]) if i > 0 else 0
        else:
            vol_ma20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(trend_4h_aligned[i]) or 
            np.isnan(squeeze_aligned[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Calculate Camarilla levels for current 1h bar using prior bar
        if i >= 1:
            # Use prior bar's OHLC for Camarilla calculation
            prev_high = high[i-1]
            prev_low = low[i-1]
            prev_close = close[i-1]
            range_ = prev_high - prev_low
            
            if range_ > 0:
                # Camarilla levels
                camarilla_h5 = prev_close + 1.1 * range_ / 12  # Resistance 5
                camarilla_h4 = prev_close + 1.1 * range_ / 6   # Resistance 4
                camarilla_h3 = prev_close + 1.1 * range_ / 4   # Resistance 3
                camarilla_l3 = prev_close - 1.1 * range_ / 4   # Support 3
                camarilla_l4 = prev_close - 1.1 * range_ / 6   # Support 4
                camarilla_l5 = prev_close - 1.1 * range_ / 12  # Support 5
            else:
                camarilla_h5 = camarilla_h4 = camarilla_h3 = prev_close
                camarilla_l3 = camarilla_l4 = camarilla_l5 = prev_close
        else:
            # Not enough data, skip
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: 4h uptrend + 1d squeeze + price breaks above H4 with volume
            if (trend_4h_aligned[i] and 
                squeeze_aligned[i] and 
                close[i] > camarilla_h4 and 
                volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = 0.20
                position = 1
            # Short: 4h downtrend + 1d squeeze + price breaks below L4 with volume
            elif (not trend_4h_aligned[i] and 
                  squeeze_aligned[i] and 
                  close[i] < camarilla_l4 and 
                  volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: trend breaks down or price reaches H5 (take profit)
            if (not trend_4h_aligned[i] or close[i] >= camarilla_h5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: trend breaks up or price reaches L5 (take profit)
            if (trend_4h_aligned[i] or close[i] <= camarilla_l5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals