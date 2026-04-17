#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla R1/S1 breakout with 12h EMA34 trend filter and volume confirmation
- Uses 6h as primary timeframe to reduce trade frequency vs lower TFs
- Camarilla pivot levels (R1/S1) from 1d provide institutional support/resistance
- Breakout above R1 with volume confirmation and 12h EMA34 uptrend = long
- Breakdown below S1 with volume confirmation and 12h EMA34 downtrend = short
- Volume confirmation (>1.5x 20-period average) filters weak breakouts
- Discrete position sizing (0.25) to minimize fee churn
- Target: 12-25 trades/year per symbol (~50-100 total over 4 years)
- Works in bull markets (captures breakouts) and bear markets (short breakdowns with trend filter)
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
    open_time = prices['open_time'].values
    
    # Get 1d data for Camarilla pivot calculation (HTF)
    df_1d = get_htf_data(prices, '1d')
    
    # Get 12h data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate Camarilla pivot levels (R1, S1) from previous day's OHLC
    # Typical price = (high + low + close) / 3
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    range_hl = df_1d['high'] - df_1d['low']
    
    # Camarilla levels
    r1 = typical_price + range_hl * 1.1 / 12
    s1 = typical_price - range_hl * 1.1 / 12
    
    # Calculate EMA34 on 12h for trend filter
    ema34_12h = pd.Series(df_12h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume average (20-period) on 6h (primary timeframe)
    volume_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1.values)
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    volume_ma_aligned = align_htf_to_ltf(prices, prices, volume_ma_6h)  # already LTF
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema34_12h_aligned[i]) or np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_trend = ema34_12h_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Look for breakouts with volume confirmation and trend alignment
            # Long: price breaks above R1 + volume spike + price > 12h EMA34 (uptrend)
            if price > r1_val and vol > 1.5 * vol_ma and price > ema_trend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + volume spike + price < 12h EMA34 (downtrend)
            elif price < s1_val and vol > 1.5 * vol_ma and price < ema_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit conditions for long
            exit_signal = False
            
            # Exit 1: Price retrace below S1 (failed breakout)
            if price < s1_val:
                exit_signal = True
            
            # Exit 2: Trend reversal (price < 12h EMA34)
            elif price < ema_trend:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions for short
            exit_signal = False
            
            # Exit 1: Price retrace above R1 (failed breakdown)
            if price > r1_val:
                exit_signal = True
            
            # Exit 2: Trend reversal (price > 12h EMA34)
            elif price > ema_trend:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R1S1_12hEMA34_Volume"
timeframe = "6h"
leverage = 1.0