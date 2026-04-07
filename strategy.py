#!/usr/bin/env python3
"""
12h_camarilla_pivot_1d_trend_volume_v1
Hypothesis: On 12h timeframe, use daily Camarilla pivot levels for mean-reversion entries in ranging markets, with 1d EMA for trend filter and volume confirmation to avoid false signals. Enter long when price touches S3 level in uptrend with volume spike; enter short when price touches R3 level in downtrend with volume spike. Exit when price reaches opposite pivot level or CCI crosses zero. This strategy targets mean-reversion in ranging markets while avoiding counter-trend moves, working in both bull and bear via trend filter. Uses tight entry conditions to limit trades and reduce fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_trend_volume_v1"
timeframe = "12h"
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
    
    # 1d data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values (shifted by 1 for lookback)
    ph = np.roll(high_1d, 1)
    pl = np.roll(low_1d, 1)
    pc = np.roll(close_1d, 1)
    ph[0] = ph[1] if len(ph) > 1 else 0
    pl[0] = pl[1] if len(pl) > 1 else 0
    pc[0] = pc[1] if len(pc) > 1 else 0
    
    # Camarilla levels
    range_ = ph - pl
    s1 = pc + (range_ * 1.1 / 12)
    s2 = pc + (range_ * 1.1 / 6)
    s3 = pc + (range_ * 1.1 / 4)
    r1 = pc - (range_ * 1.1 / 12)
    r2 = pc - (range_ * 1.1 / 6)
    r3 = pc - (range_ * 1.1 / 4)
    
    # 1d EMA for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    
    # Align indicators to 12h timeframe
    s3_12h = align_htf_to_ltf(prices, df_1d, s3)
    r3_12h = align_htf_to_ltf(prices, df_1d, r3)
    ema_12h = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation (20-period average on 12h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(s3_12h[i]) or np.isnan(r3_12h[i]) or np.isnan(ema_12h[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period average
        vol_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # Trend direction from EMA
        uptrend = close[i] > ema_12h[i]
        downtrend = close[i] < ema_12h[i]
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit if price reaches R3 (opposite level)
            if close[i] >= r3_12h[i]:
                exit_long = True
            # Exit if trend turns down
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
            # Exit if price reaches S3 (opposite level)
            if close[i] <= s3_12h[i]:
                exit_short = True
            # Exit if trend turns up
            elif uptrend:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry conditions
            long_entry = False
            # Price touches S3 level in uptrend with volume confirmation
            if close[i] <= s3_12h[i] * 1.002:  # Allow 0.2% slippage
                if uptrend and vol_confirm:
                    long_entry = True
            
            # Short entry conditions
            short_entry = False
            # Price touches R3 level in downtrend with volume confirmation
            if close[i] >= r3_12h[i] * 0.998:  # Allow 0.2% slippage
                if downtrend and vol_confirm:
                    short_entry = True
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals