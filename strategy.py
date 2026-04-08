#4/2025-01-11
#!/usr/bin/env python3
# 12h_1w_donchian_breakout_v1
# Hypothesis: 12-hour Donchian breakout with 1-week trend filter.
# Long when price breaks above 20-period Donchian high and price above 1-week 50 EMA.
# Short when price breaks below 20-period Donchian low and price below 1-week 50 EMA.
# Exit when price crosses below/above 10-period EMA on 12h.
# Uses 12h for entry timing and 1w for trend filter to avoid counter-trend trades.
# Designed to generate ~15-30 trades/year to avoid fee decay while capturing strong trends.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_donchian_breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate 12h indicators
    # Donchian channels (20-period)
    period_donch = 20
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    for i in range(period_donch - 1, n):
        donch_high[i] = np.max(high[i - period_donch + 1:i + 1])
        donch_low[i] = np.min(low[i - period_donch + 1:i + 1])
    
    # EMA for exit (10-period)
    close_s = pd.Series(close)
    ema_10 = close_s.ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Get 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # EMA 50 on weekly
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(ema_10[i]) or np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        price = close[i]
        upper = donch_high[i]
        lower = donch_low[i]
        ema10 = ema_10[i]
        ema50_1w = ema_50_1w_aligned[i]
        
        if position == 1:  # Long
            # Exit: price crosses below 10 EMA
            if price < ema10:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price crosses above 10 EMA
            if price > ema10:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry conditions: Donchian breakout with 1-week EMA filter
            # Bullish: price breaks above Donchian high and above weekly EMA50
            if price > upper and price > ema50_1w:
                position = 1
                signals[i] = 0.25
            # Bearish: price breaks below Donchian low and below weekly EMA50
            elif price < lower and price < ema50_1w:
                position = -1
                signals[i] = -0.25
    
    return signals