#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Candlestick Pattern Reversal with 1d Volume Filter
# Uses hammer/shooting star patterns for reversal signals - effective in both bull/bear markets
# 1d volume spike confirms institutional participation in the reversal
# 12h EMA (20) provides trend context to avoid counter-trend trades
# Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Load 1d data ONCE before loop for volume filter and trend
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d volume average (20-period) for spike detection
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # Calculate 12h EMA (20) for trend context
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 20  # for EMA and volume average
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(vol_avg_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Candlestick data for current bar
        o = open_price[i]
        h = high[i]
        l = low[i]
        c = close[i]
        v = volume[i]
        
        body = abs(c - o)
        upper_shadow = h - max(c, o)
        lower_shadow = min(c, o) - l
        
        # Hammer pattern: small body, long lower shadow, little upper shadow
        is_hammer = (body > 0 and 
                    lower_shadow > 2 * body and 
                    upper_shadow < 0.1 * body)
        
        # Shooting star pattern: small body, long upper shadow, little lower shadow
        is_shooting_star = (body > 0 and 
                           upper_shadow > 2 * body and 
                           lower_shadow < 0.1 * body)
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_spike = v > 1.5 * vol_avg_1d_aligned[i]
        
        # Trend filter: price above/below 12h EMA
        above_ema = c > ema_12h_aligned[i]
        
        if position == 0:
            # Long: hammer pattern with volume spike in downtrend context
            if is_hammer and volume_spike and not above_ema:
                position = 1
                signals[i] = position_size
            # Short: shooting star pattern with volume spike in uptrend context
            elif is_shooting_star and volume_spike and above_ema:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: shooting star pattern or trend reversal
            if is_shooting_star or c < ema_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: hammer pattern or trend reversal
            if is_hammer or c > ema_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Candlestick_Reversal_1dVolume"
timeframe = "12h"
leverage = 1.0