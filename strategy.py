#!/usr/bin/env python3
"""
1d_weekly_pivot_reversion_v1
Hypothesis: On 1d timeframe, trade reversals from weekly pivot support/resistance levels with volume confirmation and weekly trend filter. Go long when price touches weekly S1 with bullish weekly trend and volume spike; go short when price touches weekly R1 with bearish weekly trend and volume spike. Exit when price reaches opposite pivot level or weekly trend reverses. This strategy captures mean reversion in ranging markets and avoids trend-following whipsaws in strong trends. Weekly pivot levels provide institutional reference points, and volume confirmation ensures participation. Designed for low trade frequency (<25/year) to minimize fee drag and work in both bull/bear regimes via trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_pivot_reversion_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for pivot calculation and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's data)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot point
    pp_1w = (high_1w + low_1w + close_1w) / 3
    # Support and resistance levels
    s1_1w = 2 * pp_1w - high_1w
    r1_1w = 2 * pp_1w - low_1w
    
    # Weekly EMA for trend filter
    ema_1w = pd.Series(close_1w).ewm(span=20, adjust=False).mean().values
    
    # Align weekly data to daily timeframe
    pp_1w_1d = align_htf_to_ltf(prices, df_1w, pp_1w)
    s1_1w_1d = align_htf_to_ltf(prices, df_1w, s1_1w)
    r1_1w_1d = align_htf_to_ltf(prices, df_1w, r1_1w)
    ema_1w_1d = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume confirmation (20-day average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(pp_1w_1d[i]) or np.isnan(s1_1w_1d[i]) or np.isnan(r1_1w_1d[i]) or
            np.isnan(ema_1w_1d[i]) or np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x 20-day average
        vol_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # Weekly trend direction
        uptrend = close[i] > ema_1w_1d[i]
        downtrend = close[i] < ema_1w_1d[i]
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit if price reaches R1 (opposite pivot level)
            if high[i] >= r1_1w_1d[i]:
                exit_long = True
            # Exit if weekly trend turns bearish
            elif downtrend:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit if price reaches S1 (opposite pivot level)
            if low[i] <= s1_1w_1d[i]:
                exit_short = True
            # Exit if weekly trend turns bullish
            elif uptrend:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price touches S1 with bullish weekly trend and volume confirmation
            long_entry = False
            if low[i] <= s1_1w_1d[i] * 1.002:  # Allow 0.2% tolerance for touch
                if uptrend and vol_confirm:
                    long_entry = True
            
            # Short entry: price touches R1 with bearish weekly trend and volume confirmation
            short_entry = False
            if high[i] >= r1_1w_1d[i] * 0.998:  # Allow 0.2% tolerance for touch
                if downtrend and vol_confirm:
                    short_entry = True
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals