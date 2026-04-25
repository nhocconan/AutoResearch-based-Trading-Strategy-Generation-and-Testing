#!/usr/bin/env python3
"""
1h_Camarilla_H3L3_Breakout_1dTrend_VolumeSpike
Hypothesis: On 1h timeframe, use daily Camarilla H3/L3 levels as key support/resistance. 
Break above H3 with volume spike and daily uptrend (close > EMA34) = long signal. 
Break below L3 with volume spike and daily downtrend (close < EMA34) = short signal.
Use 4h EMA50 as secondary trend filter for stronger confirmation. 
Fixed position size 0.20 to limit drawdown. Session filter 08-20 UTC to avoid low-liquidity hours.
Designed to capture institutional breakout moves while avoiding chop. Works in bull/bear via trend filters.
Target: 15-35 trades/year per symbol (~60-140 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for efficiency
    hours = prices.index.hour
    
    # 1d data for Camarilla levels and trend filter (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous 1d bar
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    camarilla_h3 = c_1d + (h_1d - l_1d) * 1.1 / 4
    camarilla_l3 = c_1d - (h_1d - l_1d) * 1.1 / 4
    
    # 1d EMA34 for trend filter (loaded ONCE)
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # 4h EMA50 for secondary trend confirmation (loaded ONCE)
    df_4h = get_htf_data(prices, '4h')
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to LTF (1h)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1h volume spike: current volume > 2.0 * 20-period volume MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need volume MA (20) + aligned HTF arrays
    start_idx = max(20, 0)
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC only
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        if position == 0:
            # Long: price breaks above camarilla H3 with volume spike, 1d uptrend, and 4h uptrend
            long_breakout = (curr_close > camarilla_h3_aligned[i]) and vol_spike[i] and \
                           (curr_close > ema_34_1d_aligned[i]) and (curr_close > ema_50_4h_aligned[i])
            # Short: price breaks below camarilla L3 with volume spike, 1d downtrend, and 4h downtrend
            short_breakout = (curr_close < camarilla_l3_aligned[i]) and vol_spike[i] and \
                            (curr_close < ema_34_1d_aligned[i]) and (curr_close < ema_50_4h_aligned[i])
            
            if long_breakout:
                signals[i] = 0.20
                position = 1
            elif short_breakout:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.20
            # Exit: price breaks below camarilla L3 OR 1d trend turns down OR 4h trend turns down
            if (curr_close < camarilla_l3_aligned[i]) or (curr_close < ema_34_1d_aligned[i]) or (curr_close < ema_50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.20
            # Exit: price breaks above camarilla H3 OR 1d trend turns up OR 4h trend turns up
            if (curr_close > camarilla_h3_aligned[i]) or (curr_close > ema_34_1d_aligned[i]) or (curr_close > ema_50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_H3L3_Breakout_1dTrend_VolumeSpike"
timeframe = "1h"
leverage = 1.0