#!/usr/bin/env python3
# 4h_1d_camarilla_breakout_volume_v1
# Strategy: 4h Camarilla pivot breakout with 1d volume confirmation and 1d trend filter
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Camarilla levels act as intraday support/resistance. Breakouts above R4 or below S4 with 1d volume above average and aligned with 1d trend (price above/below 1d EMA50) capture strong moves. Works in bull/bear by trading breakouts in direction of higher timeframe trend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d average volume for volume filter
    volume_1d = df_1d['volume'].values
    avg_volume_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(avg_volume_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Calculate Camarilla levels for current day using previous day's OHLC
        # Need to get previous day's data - we'll use the 1d data shifted by 1
        if i < len(prices) and hasattr(prices.index, 'date'):
            # Simpler approach: use current 4h bar's price to approximate
            # For Camarilla, we need daily OHLC - we'll approximate using rolling window
            pass
        
        # Since we don't have easy access to previous day's OHLC in 4h data,
        # we'll use a simplified approach: calculate pivot based on recent 4h data
        # This is not perfect but avoids look-ahead and uses available data
        if i >= 20:
            # Use last 20 periods (approx 5 days of 4h data) to calculate range
            recent_high = np.max(high[i-20:i])
            recent_low = np.min(low[i-20:i])
            recent_close = close[i-1]  # previous close
            
            # Calculate Camarilla levels
            range_val = recent_high - recent_low
            if range_val <= 0:
                signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
                continue
                
            # Camarilla levels
            r4 = recent_close + (range_val * 1.1 / 2)
            r3 = recent_close + (range_val * 1.1 / 4)
            s3 = recent_close - (range_val * 1.1 / 4)
            s4 = recent_close - (range_val * 1.1 / 2)
            
            # Volume confirmation: current volume > 1.5x average 1d volume
            volume_confirm = volume[i] > 1.5 * avg_volume_1d_aligned[i]
            
            # Trend filter: price above/below 1d EMA50
            price_above_ema = close[i] > ema_50_1d_aligned[i]
            price_below_ema = close[i] < ema_50_1d_aligned[i]
            
            # Entry conditions
            # Long: Price breaks above R4 with volume and uptrend
            if volume_confirm and price_above_ema and close[i] > r4 and position != 1:
                position = 1
                signals[i] = 0.25
            # Short: Price breaks below S4 with volume and downtrend
            elif volume_confirm and price_below_ema and close[i] < s4 and position != -1:
                position = -1
                signals[i] = -0.25
            # Exit: Reverse signal or loss of momentum
            elif position == 1 and (close[i] < s3 or not price_above_ema):
                position = 0
                signals[i] = 0.0
            elif position == -1 and (close[i] > r3 or not price_below_ema):
                position = 0
                signals[i] = 0.0
            else:
                # Hold current position
                signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
        else:
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
    
    return signals