#!/usr/bin/env python3
name = "12h_WVWAP_Trend_Filter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1W DATA FOR TREND FILTER ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # 20-period EMA on weekly close
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_12h = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # === 1D DATA FOR VWAP AND VOLUME ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # VWAP = typical price * volume / cumulative volume
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    pv_1d = typical_price_1d * volume_1d
    cum_pv = np.nancumsum(pv_1d)
    cum_vol = np.nancumsum(volume_1d)
    vwap_1d = np.divide(cum_pv, cum_vol, out=np.zeros_like(cum_pv), where=cum_vol!=0)
    vwap_1d_12h = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # 20-period volume average on 1d
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_12h = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (vol_ma_1d_12h * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(60, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema20_1w_12h[i]) or np.isnan(vwap_1d_12h[i]) or 
            np.isnan(vol_ma_1d_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price above VWAP (intraday strength) + above weekly EMA (trend) + volume spike
            if (close[i] > vwap_1d_12h[i] and 
                close[i] > ema20_1w_12h[i] and
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price below VWAP + below weekly EMA + volume spike
            elif (close[i] < vwap_1d_12h[i] and 
                  close[i] < ema20_1w_12h[i] and
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price breaks below VWAP OR below weekly EMA
            if close[i] < vwap_1d_12h[i] or close[i] < ema20_1w_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above VWAP OR above weekly EMA
            if close[i] > vwap_1d_12h[i] or close[i] > ema20_1w_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals