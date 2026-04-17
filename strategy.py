# Your turn. Remember: NO REASONING, no explanation, just the code. Let's go.  
#!/usr/bin/env python3
"""
Hypothesis: Weekly price extremes (highest high and lowest low) act as strong support/resistance.
On 6h timeframe, price breaking above the weekly high with volume confirmation indicates
continuation of the weekly uptrend, while breaking below the weekly low indicates
continuation of the weekly downtrend. Entries are filtered by 1d EMA50 to ensure alignment
with the daily trend. Exits occur when price returns to the weekly midpoint, reducing
exposure during reversals. Designed for 6h to capture multi-day trends with low frequency.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for high/low extremes and midpoint
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly highest high and lowest low
    whigh = df_1w['high'].values
    wlow = df_1w['low'].values
    
    # Weekly midpoint for exit
    wmid = (whigh + wlow) / 2
    
    # Get daily data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    dclose = df_1d['close'].values
    ema_50 = pd.Series(dclose).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly and daily levels to 6h timeframe
    whigh_6h = align_htf_to_ltf(prices, df_1w, whigh)
    wlow_6h = align_htf_to_ltf(prices, df_1w, wlow)
    wmid_6h = align_htf_to_ltf(prices, df_1w, wmid)
    ema_50_6h = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: 24-period volume MA on 6h (4 days)
    volume_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 60  # warmup for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(whigh_6h[i]) or np.isnan(wlow_6h[i]) or np.isnan(wmid_6h[i]) or
            np.isnan(ema_50_6h[i]) or np.isnan(volume_ma_24.iloc[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_24.iloc[i]
        
        if position == 0:
            # Long: break above weekly high with volume and above daily EMA50
            if price > whigh_6h[i] and vol > 2.0 * vol_ma and price > ema_50_6h[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below weekly low with volume and below daily EMA50
            elif price < wlow_6h[i] and vol > 2.0 * vol_ma and price < ema_50_6h[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to weekly midpoint
            if price < wmid_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to weekly midpoint
            if price > wmid_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyHighLow_Volume_EMA50"
timeframe = "6h"
leverage = 1.0