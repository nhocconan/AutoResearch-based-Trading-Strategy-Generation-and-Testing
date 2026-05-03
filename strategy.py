#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R mean reversion with 1w EMA50 trend filter and volume confirmation
# Long when Williams %R < -80 (oversold), close > 1w EMA50, volume > 2.0x 24-bar average
# Short when Williams %R > -20 (overbought), close < 1w EMA50, volume > 2.0x 24-bar average
# Uses Williams %R for momentum exhaustion, 1w EMA50 for trend filter, volume for confirmation
# Designed for low trade frequency (~12-37/year on 12h) to minimize fee drag
# Works in bull (buy oversold dips in uptrend) and bear (sell overbought rallies in downtrend)

name = "12h_WilliamsR_Volume_1wEMA50_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Williams %R (14-period) on 1w timeframe
    highest_high_1w = pd.Series(df_1w['high'].values).rolling(window=14, min_periods=14).max().values
    lowest_low_1w = pd.Series(df_1w['low'].values).rolling(window=14, min_periods=14).min().values
    close_1w_series = df_1w['close'].values
    williams_r = -100 * (highest_high_1w - close_1w_series) / (highest_high_1w - lowest_low_1w)
    williams_r_aligned = align_htf_to_ltf(prices, df_1w, williams_r)
    
    # Volume confirmation (2.0x 24-period average on 12h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = max(50, 14, 24) + 1  # EMA50(1w) + Williams %R(14) + volume MA(24) + shift(1)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_50_aligned[i]) or np.isnan(williams_r_aligned[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Williams %R < -80 (oversold), close > 1w EMA50, volume spike
            if (williams_r_aligned[i] < -80 and 
                close[i] > ema_50_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R > -20 (overbought), close < 1w EMA50, volume spike
            elif (williams_r_aligned[i] > -20 and 
                  close[i] < ema_50_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R > -20 (overbought) or close < 1w EMA50 (trend failure)
            if (williams_r_aligned[i] > -20 or 
                close[i] < ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R < -80 (oversold) or close > 1w EMA50 (trend failure)
            if (williams_r_aligned[i] < -80 or 
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals