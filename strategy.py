#!/usr/bin/env python3
"""
4h_CCI_Trend_Filter_V1
Strategy: 4h Commodity Channel Index (CCI) with daily EMA34 trend filter and volume confirmation.
Long: CCI(20) > -100 + price > daily EMA34 + volume > 1.3x 20-period average
Short: CCI(20) < 100 + price < daily EMA34 + volume > 1.3x 20-period average
Exit: Opposite condition or trend reversal
Position size: 0.25
Designed to capture mean-reversion in ranging markets while respecting trend.
Timeframe: 4h
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
    
    # Calculate CCI(20)
    def cci(high, low, close, window=20):
        n = len(high)
        cci_out = np.full(n, np.nan)
        if n < window:
            return cci_out
        
        # Typical Price
        tp = (high + low + close) / 3
        
        # Moving Average of TP
        tp_ma = pd.Series(tp).rolling(window=window, min_periods=window).mean().values
        
        # Mean Deviation
        md = np.zeros(n)
        for i in range(window-1, n):
            md[i] = np.mean(np.abs(tp[i-window+1:i+1] - tp_ma[i]))
        
        # CCI
        for i in range(window-1, n):
            if md[i] > 0:
                cci_out[i] = (tp[i] - tp_ma[i]) / (0.015 * md[i])
            else:
                cci_out[i] = 0
        
        return cci_out
    
    cci_val = cci(high, low, close, 20)
    
    # Calculate daily EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 4h volume average (20-period)
    df_4h = get_htf_data(prices, '4h')
    volume_4h = df_4h['volume'].values
    volume_ma20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_ma20_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_ma20_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    for i in range(50, n):  # warmup for indicators
        # Session filter: 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        # Skip if any required data is not available
        if (np.isnan(cci_val[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_ma20_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current 4h volume
        vol_4h_current = align_htf_to_ltf(prices, df_4h, volume_4h)[i]
        volume_filter = vol_4h_current > (1.3 * volume_ma20_4h_aligned[i])
        
        # Trend filter: price above/below daily EMA34
        trend_up = close[i] > ema_34_1d_aligned[i]
        trend_down = close[i] < ema_34_1d_aligned[i]
        
        # Entry signals
        if position == 0:
            # Long: CCI > -100 (not deeply oversold) + volume filter + trend up
            if cci_val[i] > -100 and volume_filter and trend_up:
                signals[i] = 0.25
                position = 1
            # Short: CCI < 100 (not deeply overbought) + volume filter + trend down
            elif cci_val[i] < 100 and volume_filter and trend_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: CCI < -100 (deeply oversold) or trend down
            if cci_val[i] < -100 or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: CCI > 100 (deeply overbought) or trend up
            if cci_val[i] > 100 or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_CCI_Trend_Filter_V1"
timeframe = "4h"
leverage = 1.0