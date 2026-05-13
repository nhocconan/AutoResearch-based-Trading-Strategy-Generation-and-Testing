#/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend
Hypothesis: On 1h timeframe, breakouts above 4h Camarilla R1 or below S1 with volume confirmation 
and aligned 4h trend (close > EMA20) capture momentum with controlled frequency. 
Uses 0.20 position size and session filter (08-20 UTC) to target 15-35 trades/year.
Works in bull via breakout continuation, bear via breakdown continuation.
"""

name = "1h_Camarilla_R1_S1_Breakout_4hTrend"
timeframe = "1h"
leverage = 1.0

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
    
    # Session filter: 08-20 UTC (pre-compute)
    hours = prices.index.hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Camarilla pivots and trend filter (once before loop)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate Camarilla pivot levels for 4h (using previous bar)
    prev_close = df_4h['close'].shift(1).values
    prev_high = df_4h['high'].shift(1).values
    prev_low = df_4h['low'].shift(1).values
    
    # R1 = C + ((H-L) * 1.1/6)
    # S1 = C - ((H-L) * 1.1/6)
    camarilla_r1 = prev_close + ((prev_high - prev_low) * 1.1 / 6)
    camarilla_s1 = prev_close - ((prev_high - prev_low) * 1.1 / 6)
    
    # Align Camarilla levels to 1h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    
    # 4h trend filter: EMA(20) on close
    ema20_4h = pd.Series(df_4h['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # Volume confirmation: current volume > 1.5x 24-period average (1 day on 1h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(24, n):  # Start after warmup
        if not session_mask[i]:
            signals[i] = 0.0
            continue
            
        if position == 0:
            # LONG: Breakout above R1 with volume confirmation and uptrend
            if (close[i] > camarilla_r1_aligned[i] and 
                volume_filter[i] and 
                close[i] > ema20_4h_aligned[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Breakdown below S1 with volume confirmation and downtrend
            elif (close[i] < camarilla_s1_aligned[i] and 
                  volume_filter[i] and 
                  close[i] < ema20_4h_aligned[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below R1 or trend reverses
            if (close[i] < camarilla_r1_aligned[i]) or \
               (close[i] < ema20_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price re-enters above S1 or trend reverses
            if (close[i] > camarilla_s1_aligned[i]) or \
               (close[i] > ema20_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals