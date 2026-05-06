#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d VWAP with Bollinger Bands for mean-reversion trades
# - Long when price touches lower Bollinger Band (20, 2) from 1d VWAP with volume spike and above 1d EMA50
# - Short when price touches upper Bollinger Band (20, 2) from 1d VWAP with volume spike and below 1d EMA50
# - Exit when price crosses back above/below 1d VWAP
# - Uses Bollinger Bands calculated from 1d VWAP to capture institutional support/resistance
# - Volume filter requires current volume > 1.5x 20-period average to confirm institutional interest
# - EMA50 trend filter ensures trades align with higher timeframe trend
# - Designed for institutional mean-reversion at key daily levels with volume confirmation
# - Target: 75-200 total trades over 4 years (19-50/year) with 0.25 position sizing

name = "4h_VWAP_BBands_1dEMA50_Volume"
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
    
    # Get 1d data for VWAP and EMA calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d typical price and VWAP
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    pv = (typical_price * df_1d['volume']).values
    vol = df_1d['volume'].values
    
    # Calculate cumulative VWAP
    cum_pv = np.cumsum(pv)
    cum_vol = np.cumsum(vol)
    vwap = np.divide(cum_pv, cum_vol, out=np.zeros_like(cum_pv), where=cum_vol!=0)
    
    # Calculate 1d Bollinger Bands (20, 2) from VWAP
    vwap_series = pd.Series(vwap)
    vwap_ma = vwap_series.rolling(window=20, min_periods=20).mean().values
    vwap_std = vwap_series.rolling(window=20, min_periods=20).std().values
    bb_lower = vwap_ma - (2 * vwap_std)
    bb_upper = vwap_ma + (2 * vwap_std)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d indicators to 4h timeframe
    bb_lower_4h = align_htf_to_ltf(prices, df_1d, bb_lower)
    bb_upper_4h = align_htf_to_ltf(prices, df_1d, bb_upper)
    vwap_4h = align_htf_to_ltf(prices, df_1d, vwap)
    ema_50_1d_4h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filters (4h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)  # Volume confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if any critical value is NaN
        if (np.isnan(bb_lower_4h[i]) or np.isnan(bb_upper_4h[i]) or 
            np.isnan(vwap_4h[i]) or np.isnan(ema_50_1d_4h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price touches or goes below BB lower with volume spike and above EMA50
            if low[i] <= bb_lower_4h[i] and volume_spike[i] and close[i] > ema_50_1d_4h[i]:
                signals[i] = 0.25
                position = 1
            # Short: price touches or goes above BB upper with volume spike and below EMA50
            elif high[i] >= bb_upper_4h[i] and volume_spike[i] and close[i] < ema_50_1d_4h[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses above VWAP
            if close[i] > vwap_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses below VWAP
            if close[i] < vwap_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals