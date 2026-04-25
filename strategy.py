#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_4hTrend_VolumeSpike
Hypothesis: Camarilla R1/S1 levels on 1h act as intraday support/resistance. A break above R1 with 4h uptrend and volume spike signals long; break below S1 with 4h downtrend and volume spike signals short. Uses 4h for trend direction and 1h for entry timing. Session filter (08-20 UTC) reduces noise. Target: 15-37 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # 1h data for Camarilla calculation
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h data for trend filter (loaded ONCE)
    df_4h = get_htf_data(prices, '4h')
    
    # 4h EMA34 for trend filter
    ema_34_4h = pd.Series(df_4h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime64 arithmetic in loop
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Calculate Camarilla levels for each 1h bar using prior bar's range
    # H1, L1, C1 = high, low, close of previous 1h bar
    h1 = np.roll(high, 1)
    l1 = np.roll(low, 1)
    c1 = np.roll(close, 1)
    # Set first bar to NaN (no prior bar)
    h1[0] = np.nan
    l1[0] = np.nan
    c1[0] = np.nan
    
    # Camarilla R1 and S1
    r1 = c1 + (1.1/12) * (h1 - l1)
    s1 = c1 - (1.1/12) * (h1 - l1)
    
    # 1h volume spike: current volume > 2.0 * 20-period volume MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma_20)
    
    # Start index: need roll(1) + vol_ma_20 + aligned HTF
    start_idx = max(20, 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(r1[i]) or np.isnan(s1[i]) or 
            np.isnan(ema_34_4h_aligned[i]) or np.isnan(vol_ma_20[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume spike and 4h uptrend
            long_breakout = (curr_close > r1[i]) and vol_spike[i] and (curr_close > ema_34_4h_aligned[i])
            # Short: price breaks below S1 with volume spike and 4h downtrend
            short_breakout = (curr_close < s1[i]) and vol_spike[i] and (curr_close < ema_34_4h_aligned[i])
            
            if long_breakout:
                signals[i] = 0.20
                position = 1
                entry_price = curr_close
            elif short_breakout:
                signals[i] = -0.20
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.20
            # Exit: price breaks below S1 OR trend turns down
            if (curr_close < s1[i]) or (curr_close < ema_34_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.20
            # Exit: price breaks above R1 OR trend turns up
            if (curr_close > r1[i]) or (curr_close > ema_34_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hTrend_VolumeSpike"
timeframe = "1h"
leverage = 1.0