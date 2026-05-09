#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d long/short at weekly VWAP with 1w EMA trend filter and volume spike.
# Uses weekly VWAP as dynamic support/resistance and EMA for trend direction.
# Designed to work in both bull (pullbacks to VWAP) and bear (rejections at VWAP).
# Target: 10-25 trades/year to avoid fee drag.
name = "1d_VWAP_Reversal_1wTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 10:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for VWAP and EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly VWAP (typical price * volume) / cumulative volume
    typical_price = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3.0
    vwap_numerator = (typical_price * df_1w['volume']).cumsum()
    vwap_denominator = df_1w['volume'].cumsum()
    vwap = (vwap_numerator / vwap_denominator).values
    
    # Weekly EMA(34) for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align VWAP and EMA to daily timeframe (use previous week's values)
    vwap_aligned = align_htf_to_ltf(prices, df_1w, vwap)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation: volume > 2.0x 20-period EMA (high threshold for fewer trades)
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (2.0 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 1  # Need at least 1 week of data for VWAP/EMA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(vwap_aligned[i]) or np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Enter long: price crosses above VWAP + 1w uptrend + volume spike
            if (price > vwap_aligned[i] and price > ema_34_1w_aligned[i] and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price crosses below VWAP + 1w downtrend + volume spike
            elif (price < vwap_aligned[i] and price < ema_34_1w_aligned[i] and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below VWAP or trend reverses
            if price < vwap_aligned[i] or price < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above VWAP or trend reverses
            if price > vwap_aligned[i] or price > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals