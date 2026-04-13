#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Williams %R extreme with 1d trend filter and volume confirmation
    # Long: Williams %R < -80 (oversold) AND volume > 1.5x 20-period average AND price > 1d EMA200
    # Short: Williams %R > -20 (overbought) AND volume > 1.5x 20-period average AND price < 1d EMA200
    # Exit: Williams %R returns to -50 (mean reversion)
    # Using 1d for EMA200 trend filter (major trend), 6h for Williams %R timing
    # Discrete position sizing (0.25) to balance return and drawdown
    # Target: 12-37 trades/year (~50-150 over 4 years) to minimize fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA200 (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 6h Williams %R (14-period)
    lookback = 14
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(lookback - 1, n):
        highest_high[i] = np.max(high[i - lookback + 1:i + 1])
        lowest_low[i] = np.min(low[i - lookback + 1:i + 1])
    
    williams_r = np.full(n, np.nan)
    for i in range(lookback - 1, n):
        if highest_high[i] != lowest_low[i]:
            williams_r[i] = (highest_high[i] - close[i]) / (highest_high[i] - lowest_low[i]) * -100
        else:
            williams_r[i] = -50
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Trend filter: only long if price > 1d EMA200, only short if price < 1d EMA200
        long_trend_ok = close[i] > ema_1d_aligned[i]
        short_trend_ok = close[i] < ema_1d_aligned[i]
        
        # Entry logic: Williams %R extreme + volume + trend
        long_entry = (williams_r[i] < -80) and vol_confirm and long_trend_ok
        short_entry = (williams_r[i] > -20) and vol_confirm and short_trend_ok
        
        # Exit logic: Williams %R returns to -50 (mean reversion)
        long_exit = williams_r[i] > -50
        short_exit = williams_r[i] < -50
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_williamsr_extreme_volume_trend_v1"
timeframe = "6h"
leverage = 1.0