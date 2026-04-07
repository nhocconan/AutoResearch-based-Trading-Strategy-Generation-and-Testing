#!/usr/bin/env python3
"""
1d_ema_trend_1w_volume_v1
Hypothesis: On daily timeframe, use EMA(50) for trend direction and EMA(200) as long-term filter.
Enter long when EMA(50) crosses above EMA(200) AND weekly EMA(50) is rising.
Enter short when EMA(50) crosses below EMA(200) AND weekly EMA(50) is falling.
Volume confirmation: daily volume > 1.5x 20-day average.
This captures medium-term trends with institutional confirmation, reducing whipsaw.
Target: 10-20 trades/year to minimize fee drag while capturing sustained moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_ema_trend_1w_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price and volume data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    w_close = df_1w['close'].values
    
    # Calculate EMA(50) and EMA(200) on daily
    close_series = pd.Series(close)
    ema50 = close_series.ewm(span=50, adjust=False).mean().values
    ema200 = close_series.ewm(span=200, adjust=False).mean().values
    
    # Calculate EMA(50) on weekly for trend filter
    w_close_series = pd.Series(w_close)
    ema50_w = w_close_series.ewm(span=50, adjust=False).mean().values
    
    # Align weekly EMA(50) to daily timeframe
    ema50_w_aligned = align_htf_to_ltf(prices, df_1w, ema50_w)
    
    # Volume filter: daily volume > 1.5x 20-day average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean()
    vol_ratio = vol_series / vol_ma
    vol_ratio = vol_ratio.fillna(0).values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):
        # Skip if EMAs not available
        if np.isnan(ema50[i]) or np.isnan(ema200[i]) or np.isnan(ema50_w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Daily EMA crossover
        ema50_above = ema50[i] > ema200[i]
        ema50_below = ema50[i] < ema200[i]
        
        # Previous day's EMA positions for crossover detection
        ema50_above_prev = ema50[i-1] > ema200[i-1]
        ema50_below_prev = ema50[i-1] < ema200[i-1]
        
        # Weekly EMA trend
        ema50_w_rising = ema50_w_aligned[i] > ema50_w_aligned[i-1]
        ema50_w_falling = ema50_w_aligned[i] < ema50_w_aligned[i-1]
        
        # Volume confirmation
        vol_confirmed = vol_ratio[i] > 1.5
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit when EMA(50) crosses below EMA(200)
            if ema50_below and ema50_above_prev:
                exit_long = True
            # Exit when weekly EMA(50) starts falling
            elif ema50_w_falling:
                exit_long = True
            # Exit when volume drops below average
            elif vol_ratio[i] < 1.0:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit when EMA(50) crosses above EMA(200)
            if ema50_above and ema50_below_prev:
                exit_short = True
            # Exit when weekly EMA(50) starts rising
            elif ema50_w_rising:
                exit_short = True
            # Exit when volume drops below average
            elif vol_ratio[i] < 1.0:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: EMA(50) crosses above EMA(200) AND weekly EMA rising AND volume confirmed
            long_entry = ema50_above and ema50_below_prev and ema50_w_rising and vol_confirmed
            
            # Short entry: EMA(50) crosses below EMA(200) AND weekly EMA falling AND volume confirmed
            short_entry = ema50_below and ema50_above_prev and ema50_w_falling and vol_confirmed
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals