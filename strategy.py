# 12h_volatility_breakout_1d_trend_volume_v1
# Hypothesis: Use 12h volatility breakout above 1d ATR-based channel with volume confirmation and trend filter.
# Breakouts capture momentum in both bull and bear markets. Volume confirms institutional interest.
# 1d trend filter ensures we trade with higher timeframe momentum. Target 15-30 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_volatility_breakout_1d_trend_volume_v1"
timeframe = "12h"
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
    
    # Get 1d data for trend filter and ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d ATR(14) for volatility channel
    tr1 = high_1d - low_1d
    tr2 = np.abs(np.concatenate([[high_1d[0]], high_1d[:-1]]) - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(np.concatenate([[low_1d[0]], low_1d[:-1]]) - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # 12h Donchian channel (20-period)
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(high_max_20[i]) or np.isnan(low_min_20[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or
            np.isnan(vol_avg_20[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below 12h Donchian lower band OR trend turns bearish
            if (close[i] < low_min_20[i]) or (close[i] < ema_50_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above 12h Donchian upper band OR trend turns bullish
            if (close[i] > high_max_20[i]) or (close[i] > ema_50_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above upper band + volume confirmation + bullish trend
            if (close[i] > high_max_20[i] and 
                volume[i] > vol_avg_20[i] * 1.5 and 
                close[i] > ema_50_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below lower band + volume confirmation + bearish trend
            elif (close[i] < low_min_20[i] and 
                  volume[i] > vol_avg_20[i] * 1.5 and 
                  close[i] < ema_50_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals