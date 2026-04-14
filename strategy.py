#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h trend based on weekly EMA200 for directional bias, 
# Enter on daily Chandelier Exit reversal signals with volume confirmation.
# Weekly EMA200 provides robust long-term trend filter to avoid counter-trend trades.
# Chandelier Exit (ATR-based trailing stop) signals reversals when price closes 
# above/below the stop level, indicating momentum shifts.
# Volume > 2x average confirms institutional participation in the reversal.
# Designed for 6H timeframe to capture medium-term swings with low trade frequency.
# Works in bull/bear as weekly EMA200 adapts to long-term trend.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA(200) for trend filter
    ema_len = 200
    if len(df_1w) < ema_len:
        return np.zeros(n)
    
    ema_1w = pd.Series(df_1w['close']).ewm(span=ema_len, adjust=False, min_periods=ema_len).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Load daily data for Chandelier Exit
    df_1d = get_htf_data(prices, '1d')
    
    # Chandelier Exit calculation on daily timeframe
    atr_len = 22
    mult = 3.0
    if len(df_1d) < atr_len:
        return np.zeros(n)
    
    # True Range
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=atr_len, adjust=False, min_periods=atr_len).mean().values
    
    # Chandelier Exit for longs and shorts
    ce_long = df_1d['high'].rolling(window=atr_len, min_periods=atr_len).max() - mult * atr
    ce_short = df_1d['low'].rolling(window=atr_len, min_periods=atr_len).min() + mult * atr
    
    ce_long_aligned = align_htf_to_ltf(prices, df_1d, ce_long.values)
    ce_short_aligned = align_htf_to_ltf(prices, df_1d, ce_short.values)
    
    # Volume confirmation: 2x average volume
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(100, ema_len, atr_len, 50)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_1w_aligned[i]) or 
            np.isnan(ce_long_aligned[i]) or
            np.isnan(ce_short_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to weekly EMA200
        above_weekly_ema = close[i] > ema_1w_aligned[i]
        below_weekly_ema = close[i] < ema_1w_aligned[i]
        
        # Volume confirmation: current volume > 2x average
        volume_confirmed = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Enter long: price above weekly EMA200 + closes above Chandelier Exit long + volume
            if (above_weekly_ema and 
                close[i] > ce_long_aligned[i] and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Enter short: price below weekly EMA200 + closes below Chandelier Exit short + volume
            elif (below_weekly_ema and 
                  close[i] < ce_short_aligned[i] and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes below Chandelier Exit long or weekly EMA200
            if close[i] < ce_long_aligned[i] or close[i] < ema_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price closes above Chandelier Exit short or weekly EMA200
            if close[i] > ce_short_aligned[i] or close[i] > ema_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1w_EMA200_Chandelier_Volume_v1"
timeframe = "6h"
leverage = 1.0