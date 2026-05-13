#!/usr/bin/env python3
# Hypothesis: 12h volume-weighted average price (VWAP) deviation with 1d trend filter and volume confirmation.
# Uses deviation from 12h VWAP to identify mean-reversion opportunities in ranging markets,
# filtered by 1d EMA trend direction to avoid counter-trend trades. Volume ensures participation.
# Designed for low trade frequency (~15-25/year) to minimize fee drag on 12h timeframe.

name = "12h_VWAP_Deviation_MeanReversion"
timeframe = "12h"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 12-period VWAP for current timeframe
    typical_price = (high + low + close) / 3.0
    vwap_numerator = typical_price * volume
    vwap_denominator = volume
    
    # Cumulative sums for VWAP calculation
    cum_vwap_num = np.nancumsum(vwap_numerator)
    cum_vwap_den = np.nancumsum(vwap_denominator)
    vwap = np.where(cum_vwap_den != 0, cum_vwap_num / cum_vwap_den, np.nan)
    
    # Calculate deviation from VWAP as percentage
    vwap_deviation = (close - vwap) / vwap * 100.0
    
    # Volume filter: current volume > 20-period average
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient data
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(vwap[i]) or np.isnan(vwap_deviation[i]) or np.isnan(vol_ma20[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price deviates below VWAP (oversold) with uptrend and volume
            if (vwap_deviation[i] < -1.5 and  # 1.5% below VWAP
                close[i] > ema50_1d_aligned[i] and 
                volume_ok[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price deviates above VWAP (overbought) with downtrend and volume
            elif (vwap_deviation[i] > 1.5 and  # 1.5% above VWAP
                  close[i] < ema50_1d_aligned[i] and 
                  volume_ok[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to VWAP or trend changes
            if (vwap_deviation[i] > -0.5 or  # Back within 0.5% of VWAP
                close[i] < ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price returns to VWAP or trend changes
            if (vwap_deviation[i] < 0.5 or   # Back within 0.5% of VWAP
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals