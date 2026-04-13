#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Williams %R extreme reversal with 1w trend filter and volume confirmation
    # Long: Williams %R < -80 (oversold) AND price > 1w EMA200 (bullish trend) AND volume > 1.5x 20-period average
    # Short: Williams %R > -20 (overbought) AND price < 1w EMA200 (bearish trend) AND volume > 1.5x 20-period average
    # Exit: Williams %R crosses above -50 (for long) or below -50 (for short)
    # Using 1w for trend (structure) and 6h for Williams %R + volume (timing)
    # Discrete position sizing (0.25) to balance return and drawdown
    # Target: 12-37 trades/year (~50-150 over 4 years) to minimize fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA200 trend filter (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # Calculate 1w EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Williams %R on 6h: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Using 14-period lookback
    lookback = 14
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        highest_high[i] = np.max(high[i - lookback + 1:i + 1])
        lowest_low[i] = np.min(low[i - lookback + 1:i + 1])
    
    # Avoid division by zero
    hh_ll = highest_high - lowest_low
    williams_r = np.full(n, np.nan)
    for i in range(lookback - 1, n):
        if hh_ll[i] != 0:
            williams_r[i] = ((highest_high[i] - close[i]) / hh_ll[i]) * -100
        else:
            williams_r[i] = -50  # neutral when no range
    
    # Volume confirmation: >1.5x 20-period average (to reduce false signals)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # start after warmup period
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema_200_1w_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Trend filter: only long if price > 1w EMA200, only short if price < 1w EMA200
        long_trend_ok = close[i] > ema_200_1w_aligned[i]
        short_trend_ok = close[i] < ema_200_1w_aligned[i]
        
        # Williams %R levels
        wr = williams_r[i]
        long_entry = (wr < -80) and vol_confirm and long_trend_ok
        short_entry = (wr > -20) and vol_confirm and short_trend_ok
        
        # Exit: Williams %R crosses -50 (mean reversion signal)
        long_exit = wr > -50
        short_exit = wr < -50
        
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

name = "6h_1w_williamsr_extreme_volume_trend_v1"
timeframe = "6h"
leverage = 1.0