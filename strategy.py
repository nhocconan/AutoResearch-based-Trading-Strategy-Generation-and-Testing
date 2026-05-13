#!/usr/bin/env python3
"""
4H_Camarilla_R1_S1_Breakout_1D_Trend_Force_v3
Hypothesis: Use Camarilla pivot levels (R1/S1) from daily candles to identify key support/resistance. Go long when price breaks above R1 with volume confirmation and daily EMA34 trend alignment; go short when price breaks below S1 with volume confirmation and daily EMA34 trend alignment. Exit when price reverses to the pivot point (PP). Uses volume spike (>1.5x 20-period average) and trend filter (EMA34) to reduce false breakouts. Designed for 4h timeframe to target 25-40 trades/year, avoiding fee drag while capturing meaningful intraday moves around institutional pivot levels.
"""

name = "4H_Camarilla_R1_S1_Breakout_1D_Trend_Force_v3"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prrices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each daily bar
    # PP = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    typical_price = (high_1d + low_1d + close_1d) / 3.0
    price_range = high_1d - low_1d
    PP = typical_price
    R1 = close_1d + price_range * 1.1 / 12.0
    S1 = close_1d - price_range * 1.1 / 12.0
    
    # Align Camarilla levels to 4h timeframe (wait for daily close)
    PP_aligned = align_htf_to_ltf(prices, df_1d, PP)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Get daily EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate volume average (20-period) for volume spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(PP_aligned[i]) or np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current volume > 1.5x 20-period average
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # LONG: Price breaks above R1 with volume spike and price above daily EMA34 (uptrend)
            if close[i] > R1_aligned[i] and vol_spike and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 with volume spike and price below daily EMA34 (downtrend)
            elif close[i] < S1_aligned[i] and vol_spike and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to or below pivot point (PP)
            if close[i] <= PP_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to or above pivot point (PP)
            if close[i] >= PP_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals