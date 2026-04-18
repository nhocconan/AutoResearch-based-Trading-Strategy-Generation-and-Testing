#!/usr/bin/env python3
"""
4h_12h_1d_RSI_CCI_Trend
Hypothesis: Combine 12h RSI (trend filter) and 1d CCI (momentum filter) with 4h price action to catch trend continuations in both bull and bear markets. RSI > 50 indicates bullish bias, RSI < 50 bearish bias. CCI > 100 signals strong upward momentum, CCI < -100 strong downward momentum. Enter long when 12h RSI > 50 AND 1d CCI > 100 AND price closes above 4h VWAP; short when 12h RSI < 50 AND 1d CCI < -100 AND price closes below 4h VWAP. Exit when RSI crosses back below/above 50. Targets 20-30 trades/year by requiring dual timeframe alignment and momentum confirmation. Uses VWAP for institutional reference point and avoids whipsaws.
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
    
    # Calculate 4h VWAP (typical price * volume) / cumulative volume
    typical_price = (high + low + close) / 3.0
    vwap_numerator = np.cumsum(typical_price * volume)
    vwap_denominator = np.cumsum(volume)
    vwap = np.divide(vwap_numerator, vwap_denominator, 
                     out=np.full_like(typical_price, np.nan), 
                     where=vwap_denominator!=0)
    
    # Get 12h data for RSI (HTF)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h RSI (14-period)
    delta = np.diff(close_12h, prepend=close_12h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.full_like(gain, np.nan)
    avg_loss = np.full_like(loss, np.nan)
    avg_gain[13] = np.mean(gain[1:14])  # first 14 periods
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, 
                   out=np.full_like(avg_gain, np.nan), 
                   where=avg_loss!=0)
    rsi_12h = 100 - (100 / (1 + rs))
    
    # Align RSI to 4h timeframe (wait for bar close)
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
    
    # Get 1d data for CCI (HTF)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d CCI (20-period)
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    sma_tp = np.full_like(typical_price_1d, np.nan)
    mad = np.full_like(typical_price_1d, np.nan)
    
    for i in range(19, len(typical_price_1d)):
        sma_tp[i] = np.mean(typical_price_1d[i-19:i+1])
        mad[i] = np.mean(np.abs(typical_price_1d[i-19:i+1] - sma_tp[i]))
    
    cci_1d = np.divide(typical_price_1d - sma_tp, 0.015 * mad, 
                       out=np.full_like(typical_price_1d, np.nan), 
                       where=mad!=0)
    
    # Align CCI to 4h timeframe (wait for bar close)
    cci_1d_aligned = align_htf_to_ltf(prices, df_1d, cci_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need VWAP and CCI warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi_12h_aligned[i]) or np.isnan(cci_1d_aligned[i]) or 
            np.isnan(vwap[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: 12h RSI > 50 (bullish bias), 1d CCI > 100 (strong up momentum), price > VWAP
            if (rsi_12h_aligned[i] > 50 and cci_1d_aligned[i] > 100 and 
                close[i] > vwap[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: 12h RSI < 50 (bearish bias), 1d CCI < -100 (strong down momentum), price < VWAP
            elif (rsi_12h_aligned[i] < 50 and cci_1d_aligned[i] < -100 and 
                  close[i] < vwap[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: RSI falls below 50 (loss of bullish bias) or price crosses below VWAP
            if (rsi_12h_aligned[i] < 50 or close[i] < vwap[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI rises above 50 (loss of bearish bias) or price crosses above VWAP
            if (rsi_12h_aligned[i] > 50 or close[i] > vwap[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_12h_1d_RSI_CCI_Trend"
timeframe = "4h"
leverage = 1.0