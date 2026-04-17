#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H3/L3 breakout + 4h EMA34 trend filter + volume spike confirmation
- Uses 4h EMA34 as HTF trend filter to align with higher timeframe momentum
- 1h Camarilla H3/L3 breakouts capture precise entry timing with proven edge
- Volume spike (2.0x 20-period MA) confirms institutional participation
- Session filter (08-20 UTC) reduces noise trades outside active hours
- Discrete position sizing (0.20) balances return and drawdown control
- Target: 15-35 trades/year per symbol (~60-140 total over 4 years)
- Works in bull markets (buying H3 breakouts in uptrend) and bear markets (selling L3 breakdowns in downtrend)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for EMA34 trend filter (HTF)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate EMA34 on 4h for trend filter
    ema34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # Calculate Camarilla levels (H3, L3) on 1h
    def camarilla_levels(high_arr, low_arr, close_arr):
        # Classic Camarilla: H3 = close + 1.1*(high-low)/6, L3 = close - 1.1*(high-low)/6
        rng = high_arr - low_arr
        H3 = close_arr + 1.1 * rng / 6.0
        L3 = close_arr - 1.1 * rng / 6.0
        return H3, L3
    
    camarilla_H3_1h, camarilla_L3_1h = camarilla_levels(high, low, close)
    
    # Volume average (20-period) on 1h
    volume_ma_1h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        if (np.isnan(ema34_4h_aligned[i]) or np.isnan(camarilla_H3_1h[i]) or 
            np.isnan(camarilla_L3_1h[i]) or np.isnan(volume_ma_1h[i])):
            signals[i] = 0.0
            continue
        
        H3 = camarilla_H3_1h[i]
        L3 = camarilla_L3_1h[i]
        ema_trend = ema34_4h_aligned[i]
        vol_ma = volume_ma_1h[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Look for breakouts with volume confirmation and trend alignment
            # Long: price breaks above H3 + volume spike + price > 4h EMA34 (uptrend)
            if price > H3 and vol > 2.0 * vol_ma and price > ema_trend:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below L3 + volume spike + price < 4h EMA34 (downtrend)
            elif price < L3 and vol > 2.0 * vol_ma and price < ema_trend:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price retracement to midpoint of Camarilla H3-L3 range
            mid_point = (H3 + L3) / 2.0
            if price < mid_point:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price retracement to midpoint of Camarilla H3-L3 range
            mid_point = (H3 + L3) / 2.0
            if price > mid_point:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_4hEMA34_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0