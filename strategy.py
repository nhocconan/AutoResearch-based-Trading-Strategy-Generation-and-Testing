#!/usr/bin/env python3
"""
6h VWAP Reversion with 1w Trend Filter and Volume Spike.
Mean reversion when price deviates from VWAP with strong trend alignment.
Long when price < VWAP - 1.5*sigma in 1w uptrend with volume spike.
Short when price > VWAP + 1.5*sigma in 1w downtrend with volume spike.
Exit when price returns to VWAP or trend changes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_vwap_reversion_1w_trend_volume"
timeframe = "6h"
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
    
    # === 1W TREND FILTER (HTF) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    one_w_close = df_1w['close'].values
    one_w_ema = pd.Series(one_w_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    one_w_ema_aligned = align_htf_to_ltf(prices, df_1w, one_w_ema)
    
    # === VWAP AND STD DEV (6H) ===
    # Typical price
    typical_price = (high + low + close) / 3.0
    # VWAP calculation
    pv = typical_price * volume
    cum_pv = np.nancumsum(pv)
    cum_vol = np.nancumsum(volume)
    vwap = np.divide(cum_pv, cum_vol, out=np.full_like(cum_pv, np.nan), where=cum_vol!=0)
    # Standard deviation of price from VWAP
    price_diff = typical_price - vwap
    # Use rolling window for volatility
    vol_window = 20
    cum_sum_diff2 = np.nancumsum(price_diff * price_diff)
    cum_sum_diff2 = np.where(cum_vol > 0, cum_sum_diff2, 0)
    variance = np.divide(cum_sum_diff2, cum_vol, out=np.full_like(cum_sum_diff2, np.nan), where=cum_vol!=0)
    std_dev = np.sqrt(variance)
    
    # === VOLUME SPIKE FILTER (6H) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_std = pd.Series(volume).rolling(window=20, min_periods=20).std().values
    vol_threshold = vol_ma + 2.0 * vol_std  # 2 sigma volume spike
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if (np.isnan(one_w_ema_aligned[i]) or np.isnan(vwap[i]) or np.isnan(std_dev[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(vol_threshold[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from 1w EMA
        uptrend = close[i] > one_w_ema_aligned[i]
        downtrend = close[i] < one_w_ema_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price returns to VWAP OR trend turns down
            if close[i] >= vwap[i] or downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to VWAP OR trend turns up
            if close[i] <= vwap[i] or uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need volume spike
            if volume[i] < vol_threshold[i]:
                signals[i] = 0.0
                continue
            
            # Entry: VWAP deviation with trend alignment
            if close[i] < vwap[i] - 1.5 * std_dev[i] and uptrend:
                # Price below VWAP in uptrend -> long (mean reversion)
                position = 1
                signals[i] = 0.25
            elif close[i] > vwap[i] + 1.5 * std_dev[i] and downtrend:
                # Price above VWAP in downtrend -> short (mean reversion)
                position = -1
                signals[i] = -0.25
    
    return signals