#!/usr/bin/env python3
"""
12h_ema_trend_1d_volume_v2
Hypothesis: On 12h timeframe, use EMA crossover (20/50) for trend signals, filtered by daily volume confirmation and price above/below EMA200.
Enter long when EMA20 crosses above EMA50, price > EMA200, and volume > 1.5x average.
Enter short when EMA20 crosses below EMA50, price < EMA200, and volume > 1.5x average.
Exit when EMA crossover reverses or price crosses EMA200.
Targets 12-37 trades/year to minimize fee decay while capturing sustained trends.
Works in bull (trend following) and bear (short signals) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_ema_trend_1d_volume_v2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # EMA20 and EMA50 on 12h
    close_s = pd.Series(close)
    ema20 = close_s.ewm(span=20, min_periods=20, adjust=False).mean().values
    ema50 = close_s.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # EMA200 on 12h for trend filter
    ema200 = close_s.ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get daily data for additional trend filter (calculate once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA50 on daily close
    daily_close = df_1d['close'].values
    daily_close_s = pd.Series(daily_close)
    ema50_1d = daily_close_s.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align to 12h timeframe (shifted by 1 day to avoid look-ahead)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):  # Start after EMA200 warmup
        # Skip if required data not available
        if (np.isnan(ema20[i]) or np.isnan(ema50[i]) or np.isnan(ema200[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] <= 0 or
            np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filters
        price_above_ema200 = close[i] > ema200[i]
        price_below_ema200 = close[i] < ema200[i]
        daily_uptrend = ema50_1d_aligned[i] > close[i]  # Simplified: price above daily EMA50
        daily_downtrend = ema50_1d_aligned[i] < close[i]  # Simplified: price below daily EMA50
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit when EMA20 crosses below EMA50
            if ema20[i] < ema50[i]:
                exit_long = True
            # Exit when price crosses below EMA200
            elif not price_above_ema200:
                exit_long = True
            # Exit when daily trend turns down
            elif not daily_uptrend:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit when EMA20 crosses above EMA50
            if ema20[i] > ema50[i]:
                exit_short = True
            # Exit when price crosses above EMA200
            elif not price_below_ema200:
                exit_short = True
            # Exit when daily trend turns up
            elif not daily_downtrend:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: EMA20 crosses above EMA50, price > EMA200, daily uptrend, volume confirmation
            ema_cross_up = ema20[i] > ema50[i] and ema20[i-1] <= ema50[i-1]
            long_entry = ema_cross_up and price_above_ema200 and daily_uptrend and vol_confirm
            
            # Short entry: EMA20 crosses below EMA50, price < EMA200, daily downtrend, volume confirmation
            ema_cross_down = ema20[i] < ema50[i] and ema20[i-1] >= ema50[i-1]
            short_entry = ema_cross_down and price_below_ema200 and daily_downtrend and vol_confirm
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals