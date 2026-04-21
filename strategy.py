#!/usr/bin/env python3
"""
1h_Camarilla_Pivot_R1_S1_Breakout_With_Volume_4hTrend
Hypothesis: On 1h timeframe, buy breakouts above previous day's R1 when 4h trend is up and volume > 1.5x average; sell breakdowns below S1 when 4h trend is down and volume > 1.5x average. Use 4h trend for direction, 1h for entry timing. Target 15-30 trades/year per symbol.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Load 4h data ONCE before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # === Daily Camarilla pivot levels ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical price
    pp = (high_1d + low_1d + close_1d) / 3
    range_hl = high_1d - low_1d
    
    # Camarilla levels
    r1 = pp + (range_hl * 1.1 / 12)
    s1 = pp - (range_hl * 1.1 / 12)
    
    # Align to 1h timeframe (previous day's levels available at open)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === 4h EMA34 for trend filter ===
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # === Volume confirmation (20-period average) ===
    vol_ma = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(35, n):  # Start after warmup
        # Skip if indicators not ready
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_4h_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        r1_level = r1_aligned[i]
        s1_level = s1_aligned[i]
        ema_trend = ema_34_4h_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        if position == 0:
            # Long: price breaks above R1 + 4h uptrend + volume
            if (price_close > r1_level and
                price_close > ema_trend and
                vol_ratio_val > 1.5):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1 + 4h downtrend + volume
            elif (price_close < s1_level and
                  price_close < ema_trend and
                  vol_ratio_val > 1.5):
                signals[i] = -0.20
                position = -1
        
        elif position != 0:
            # Exit when price reaches opposite pivot level
            if position == 1 and price_close < s1_level:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price_close > r1_level:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_Camarilla_Pivot_R1_S1_Breakout_With_Volume_4hTrend"
timeframe = "1h"
leverage = 1.0