#!/usr/bin/env python3
name = "6h_Retracement_VWAP_Deviation_12hTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h VWAP ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Typical price and VWAP
    typical_price_12h = (high_12h + low_12h + close_12h) / 3.0
    cum_vol_tp = np.nancumsum(volume_12h * typical_price_12h)
    cum_vol = np.nancumsum(volume_12h)
    vwap_12h = np.divide(cum_vol_tp, cum_vol, out=np.full_like(cum_vol_tp, np.nan), where=cum_vol!=0)
    
    # VWAP deviation (%)
    vwap_dev_12h = (close_12h - vwap_12h) / vwap_12h * 100.0
    
    # Align VWAP deviation to 6h
    vwap_dev_12h_aligned = align_htf_to_ltf(prices, df_12h, vwap_dev_12h)
    
    # === 12h Trend (EMA50) ===
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # === 6h VWAP for entry timing ===
    typical_price_6h = (high + low + close) / 3.0
    cum_vol_tp_6h = np.nancumsum(volume * typical_price_6h)
    cum_vol_6h = np.nancumsum(volume)
    vwap_6h = np.divide(cum_vol_tp_6h, cum_vol_6h, out=np.full_like(cum_vol_tp_6h, np.nan), where=cum_vol_6h!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 50)  # Ensure enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vwap_dev_12h_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i]) or
            np.isnan(vwap_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price below VWAP (oversold) but 12h trend up
            if (close[i] < vwap_6h[i] and 
                close[i] > ema50_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price above VWAP (overbought) but 12h trend down
            elif (close[i] > vwap_6h[i] and 
                  close[i] < ema50_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price crosses above VWAP or trend breaks
            if close[i] > vwap_6h[i] or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price crosses below VWAP or trend breaks
            if close[i] < vwap_6h[i] or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals