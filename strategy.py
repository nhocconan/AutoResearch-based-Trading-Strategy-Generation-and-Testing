#!/usr/bin/env python3
# 1d_keltner_breakout_1w_trend_volume_v1
# Hypothesis: Daily Keltner breakout with 1-week EMA trend filter and volume confirmation captures directional moves in both bull and bear markets. Low trade frequency (<20/year) to minimize fee drag while maintaining edge through trend alignment and volatility-based breakouts.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_keltner_breakout_1w_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to daily
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Keltner Channel on daily
    atr_period = 10
    ema_mid = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr = pd.Series(high - low).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    upper_keltner = ema_mid + (2.0 * atr)
    lower_keltner = ema_mid - (2.0 * atr)
    
    # Volume filter: daily volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(ema_mid[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(upper_keltner[i]) or np.isnan(lower_keltner[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price < EMA20 OR price < 1w EMA50
            if (close[i] < ema_mid[i]) or (close[i] < ema_50_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price > EMA20 OR price > 1w EMA50
            if (close[i] > ema_mid[i]) or (close[i] > ema_50_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price > Upper Keltner + volume + price > 1w EMA50
            if (close[i] > upper_keltner[i]) and volume_filter[i] and (close[i] > ema_50_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price < Lower Keltner + volume + price < 1w EMA50
            elif (close[i] < lower_keltner[i]) and volume_filter[i] and (close[i] < ema_50_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals