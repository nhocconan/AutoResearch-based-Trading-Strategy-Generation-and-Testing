#!/usr/bin/env python3
"""
1d_1w_Volume_Weighted_CCI_Trend_v1
Hypothesis: Use weekly CCI(20) for trend direction (above/below +100/-100) combined with 
daily volume-weighted CCI(14) for entry timing. In bull markets (weekly CCI>100), buy 
dips when daily VW-CCI crosses below -100. In bear markets (weekly CCI<-100), sell 
rallies when daily VW-CCI crosses above +100. Volume weighting reduces false signals.
Targets 15-25 trades per year to minimize fee drag. Works in both bull (follow weekly trend) 
and bear (fade against weekly extreme).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Volume_Weighted_CCI_Trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Weekly CCI(20) for trend filter
    def cci(high, low, close, length=20):
        if len(high) < length:
            return np.full(len(high), np.nan)
        tp = (high + low + close) / 3.0
        sma = np.full(len(tp), np.nan)
        for i in range(length-1, len(tp)):
            sma[i] = np.mean(tp[i-length+1:i+1])
        mad = np.full(len(tp), np.nan)
        for i in range(length-1, len(tp)):
            mad[i] = np.mean(np.abs(tp[i-length+1:i+1] - sma[i]))
        cci_val = np.full(len(tp), np.nan)
        for i in range(length-1, len(tp)):
            if mad[i] != 0:
                cci_val[i] = (tp[i] - sma[i]) / (0.015 * mad[i])
        return cci_val
    
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    weekly_cci = cci(weekly_high, weekly_low, weekly_close, 20)
    weekly_cci_aligned = align_htf_to_ltf(prices, df_1w, weekly_cci)
    
    # Daily Volume-Weighted CCI(14)
    def vwap(high, low, close, volume):
        tp = (high + low + close) / 3.0
        vol_tp = tp * volume
        cum_vol_tp = np.cumsum(vol_tp)
        cum_vol = np.cumsum(volume)
        vwap_val = np.full(len(tp), np.nan)
        for i in range(len(tp)):
            if cum_vol[i] != 0:
                vwap_val[i] = cum_vol_tp[i] / cum_vol[i]
        return vwap_val
    
    def vw_cci(high, low, close, volume, length=14):
        if len(high) < length:
            return np.full(len(high), np.nan)
        vwap_val = vwap(high, low, close, volume)
        tp = (high + low + close) / 3.0
        deviation = tp - vwap_val
        abs_dev = np.abs(deviation)
        # Calculate mean deviation using rolling window
        mean_dev = np.full(len(abs_dev), np.nan)
        for i in range(length-1, len(abs_dev)):
            mean_dev[i] = np.mean(abs_dev[i-length+1:i+1])
        vw_cci_val = np.full(len(tp), np.nan)
        for i in range(length-1, len(tp)):
            if mean_dev[i] != 0:
                vw_cci_val[i] = deviation[i] / (0.015 * mean_dev[i])
        return vw_cci_val
    
    vw_cci_daily = vw_cci(high, low, close, volume, 14)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any data invalid
        if (np.isnan(vw_cci_daily[i]) or np.isnan(weekly_cci_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Weekly trend filter: CCI > 100 = bull, CCI < -100 = bear, else neutral
        weekly_cci_val = weekly_cci_aligned[i]
        is_bull = weekly_cci_val > 100
        is_bear = weekly_cci_val < -100
        
        # Daily VW-CCI signals
        vw_cci_val = vw_cci_daily[i]
        vw_cci_prev = vw_cci_daily[i-1] if i > 0 else vw_cci_val
        
        # Bull market: buy dips (VW-CCI crosses below -100)
        long_entry = is_bull and (vw_cci_prev > -100) and (vw_cci_val <= -100)
        # Bull market: exit when VW-CCI returns to zero
        long_exit = position == 1 and vw_cci_val >= 0
        
        # Bear market: sell rallies (VW-CCI crosses above +100)
        short_entry = is_bear and (vw_cci_prev < 100) and (vw_cci_val >= 100)
        # Bear market: exit when VW-CCI returns to zero
        short_exit = position == -1 and vw_cci_val <= 0
        
        # Signal logic
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals