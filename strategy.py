#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation
    # Long: price breaks above upper Donchian(20) AND volume > 1.5x 20-period average AND price > 12h EMA50
    # Short: price breaks below lower Donchian(20) AND volume > 1.5x 20-period average AND price < 12h EMA50
    # Exit: price returns to middle of Donchian channel (mean reversion)
    # Using 12h for EMA50 trend filter (more stable than 4h) and Donchian for structure
    # Discrete position sizing (0.25) to minimize fee churn
    # Target: 15-40 trades/year (~60-160 over 4 years) to stay within fee limits
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend filter (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # 4h Donchian(20) channels
    upper_donch = np.full(n, np.nan)
    lower_donch = np.full(n, np.nan)
    middle_donch = np.full(n, np.nan)
    for i in range(20, n):
        upper_donch[i] = np.max(high[i-20:i])
        lower_donch[i] = np.min(low[i-20:i])
        middle_donch[i] = (upper_donch[i] + lower_donch[i]) / 2
    
    # Volume confirmation: >1.5x 20-period average (to reduce false signals)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(upper_donch[i]) or np.isnan(lower_donch[i]) or 
            np.isnan(middle_donch[i]) or np.isnan(ema_12h_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Trend filter: only long if price > 12h EMA50, only short if price < 12h EMA50
        long_trend_ok = close[i] > ema_12h_aligned[i]
        short_trend_ok = close[i] < ema_12h_aligned[i]
        
        # Entry logic: Donchian breakout + volume + trend
        long_entry = (close[i] > upper_donch[i]) and vol_confirm and long_trend_ok
        short_entry = (close[i] < lower_donch[i]) and vol_confirm and short_trend_ok
        
        # Exit logic: return to middle of Donchian channel (mean reversion)
        long_exit = close[i] < middle_donch[i]
        short_exit = close[i] > middle_donch[i]
        
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

name = "4h_12h_donchian_breakout_volume_trend_v1"
timeframe = "4h"
leverage = 1.0