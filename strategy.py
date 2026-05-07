# 12h_VolumeWeighted_Momentum
# Hypothesis: Capture momentum in 12h candles using volume-weighted price action (VWAP deviation) with 1d trend filter.
# Works in both bull and bear markets: VWAP deviation identifies overextended moves ready to revert,
# while 1d trend filter ensures trades align with higher timeframe momentum. Volume confirmation
# filters out low-conviction moves. Targets 12-30 trades/year on 12h timeframe.

name = "12h_VolumeWeighted_Momentum"
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
    
    # Get 1d data for trend filter (HTF as specified)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate VWAP deviation on 12h
    typical_price = (high + low + close) / 3.0
    vwap_num = np.cumsum(typical_price * volume)
    vwap_den = np.cumsum(volume)
    vwap = vwap_num / vwap_den
    vwap_deviation = (close - vwap) / vwap  # Percentage deviation from VWAP
    
    # Align 1d EMA34 to 12h timeframe
    ema_34_1d_12h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike filter on 12h (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(vwap_deviation[i]) or np.isnan(ema_34_1d_12h[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price below VWAP (oversold), above 1d EMA34 trend, volume spike
            if vwap_deviation[i] < -0.015 and close[i] > ema_34_1d_12h[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price above VWAP (overbought), below 1d EMA34 trend, volume spike
            elif vwap_deviation[i] > 0.015 and close[i] < ema_34_1d_12h[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns to VWAP or breaks below 1d EMA34
            if vwap_deviation[i] > -0.005 or close[i] < ema_34_1d_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns to VWAP or breaks above 1d EMA34
            if vwap_deviation[i] < 0.005 or close[i] > ema_34_1d_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals