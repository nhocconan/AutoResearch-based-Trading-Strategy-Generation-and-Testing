#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h long/short at 12h VWAP with 12h EMA trend filter and volume spike.
# Uses VWAP as dynamic support/resistance and EMA for trend direction.
# Designed to work in both bull (pullbacks to VWAP) and bear (rejections at VWAP).
# Target: 20-50 trades/year to avoid fee drag.
name = "4h_VWAP_Reversal_12hTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for VWAP and EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate 12-hour VWAP (typical price * volume) / cumulative volume
    typical_price = (df_12h['high'] + df_12h['low'] + df_12h['close']) / 3.0
    vwap_numerator = (typical_price * df_12h['volume']).cumsum()
    vwap_denominator = df_12h['volume'].cumsum()
    vwap = (vwap_numerator / vwap_denominator).values
    
    # 12h EMA(34) for trend filter
    ema_34_12h = pd.Series(df_12h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align VWAP and EMA to 4h timeframe (use previous 12h bar's values)
    vwap_aligned = align_htf_to_ltf(prices, df_12h, vwap)
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Volume confirmation: volume > 2.0x 20-period EMA (high threshold for fewer trades)
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (2.0 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 1  # Need at least 1 12h period of data for VWAP/EMA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(vwap_aligned[i]) or np.isnan(ema_34_12h_aligned[i]) or 
            np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Enter long: price crosses above VWAP + 12h uptrend + volume spike
            if (price > vwap_aligned[i] and price > ema_34_12h_aligned[i] and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price crosses below VWAP + 12h downtrend + volume spike
            elif (price < vwap_aligned[i] and price < ema_34_12h_aligned[i] and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below VWAP or trend reverses
            if price < vwap_aligned[i] or price < ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above VWAP or trend reverses
            if price > vwap_aligned[i] or price > ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals