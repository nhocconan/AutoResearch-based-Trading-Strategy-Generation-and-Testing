#!/usr/bin/env python3
name = "6h_VWAP_Reversion_1dTrend_VolumeSpike"
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
    
    # === 1d VWAP for mean reversion signal ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Typical price and VWAP calculation
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    cum_tpv = np.cumsum(typical_price_1d * volume_1d)
    cum_vol = np.cumsum(volume_1d)
    vwap_1d = np.divide(cum_tpv, cum_vol, out=np.zeros_like(cum_tpv), where=cum_vol!=0)
    
    # === 1d EMA50 trend filter ===
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # === 1d Volume spike filter ===
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (2.0 * vol_avg_1d)
    
    # === 6h VWAP deviation for entry timing ===
    typical_price_6h = (high + low + close) / 3.0
    cum_tpv_6h = np.cumsum(typical_price_6h * volume)
    cum_vol_6h = np.cumsum(volume)
    vwap_6h = np.divide(cum_tpv_6h, cum_vol_6h, out=np.zeros_like(cum_tpv_6h), where=cum_vol_6h!=0)
    vwap_dev_6h = (close - vwap_6h) / vwap_6h  # % deviation from 6h VWAP
    
    # Align HTF indicators
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vwap_1d_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or
            np.isnan(vol_spike_1d_aligned[i]) or
            np.isnan(vwap_dev_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price below 1d VWAP (mean reversion) + above 1d EMA50 (uptrend) + volume spike + near 6h VWAP
            if (close[i] < vwap_1d_aligned[i] and
                close[i] > ema50_1d_aligned[i] and
                vol_spike_1d_aligned[i] > 0.5 and
                vwap_dev_6h[i] > -0.005 and vwap_dev_6h[i] < 0.005):  # Within 0.5% of 6h VWAP
                signals[i] = 0.25
                position = 1
            # Short: Price above 1d VWAP (mean reversion) + below 1d EMA50 (downtrend) + volume spike + near 6h VWAP
            elif (close[i] > vwap_1d_aligned[i] and
                  close[i] < ema50_1d_aligned[i] and
                  vol_spike_1d_aligned[i] > 0.5 and
                  vwap_dev_6h[i] > -0.005 and vwap_dev_6h[i] < 0.005):  # Within 0.5% of 6h VWAP
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price crosses above 1d VWAP or below 1d EMA50
            if close[i] > vwap_1d_aligned[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price crosses below 1d VWAP or above 1d EMA50
            if close[i] < vwap_1d_aligned[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals