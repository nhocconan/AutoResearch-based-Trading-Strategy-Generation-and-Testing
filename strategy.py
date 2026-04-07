#!/usr/bin/env python3
"""
12h_trix_signal_1w_trend_volume_v1
Hypothesis: TRIX(15) on 12h captures momentum; long when TRIX > 0 and price above 1w EMA50,
short when TRIX < 0 and price below 1w EMA50. Volume confirmation ensures momentum is real.
Weekly trend filter adapts to bull/bear markets. Target: 15-25 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_trix_signal_1w_trend_volume_v1"
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
    
    # 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 for trend filter
    ema_50 = df_1w['close'].ewm(span=50, adjust=False).mean()
    
    # Align 1w EMA50 to 12h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50.values)
    
    # TRIX(15) on 12h: triple EMA of percent change
    # TRIX = EMA(EMA(EMA(close, 15), 15), 15) - previous value
    close_series = pd.Series(close)
    ema1 = close_series.ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix = ema3.pct_change(periods=1) * 100  # as percentage
    
    # Volume confirmation (20-period average on 12h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(15, n):
        # Skip if required data not available
        if (np.isnan(trix.iloc[i]) if hasattr(trix, 'iloc') else np.isnan(trix[i]) or 
            np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        trix_val = trix.iloc[i] if hasattr(trix, 'iloc') else trix[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: TRIX turns negative or price breaks below weekly EMA50
            if trix_val < 0 or close[i] < ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: TRIX turns positive or price breaks above weekly EMA50
            if trix_val > 0 or close[i] > ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: TRIX positive with volume and price above weekly EMA50
            if (trix_val > 0 and vol_confirm and 
                close[i] > ema_50_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: TRIX negative with volume and price below weekly EMA50
            elif (trix_val < 0 and vol_confirm and 
                  close[i] < ema_50_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals