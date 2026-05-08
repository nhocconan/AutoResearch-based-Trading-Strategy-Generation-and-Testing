#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h RSI trend filter and 1d volume spike for entry timing.
# Uses 4h RSI(14) > 50 for long bias, < 50 for short bias.
# Entry triggered when 1h price crosses above/below 1h VWAP with 1d volume > 2.0x 20-day average.
# Exits on RSI flip or volume normalization.
# Targets 15-35 trades per year (~60-140 total over 4 years) to minimize fee drift.
# Works in bull/bear: RSI filters trend, volume spikes capture momentum in both directions.

name = "1h_RSI4h_VWAP_Volume1d"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1h VWAP calculation
    typical_price = (high + low + close) / 3.0
    vwap_num = (typical_price * volume).cumsum()
    vwap_den = volume.cumsum()
    vwap = vwap_num / vwap_den
    
    # 4h RSI for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 15:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_4h = 100 - (100 / (1 + rs))
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # 1d volume for spike filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = vol_1d / (vol_ma_1d + 1e-10)
    vol_spike_1d = vol_ratio_1d > 2.0
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(vwap[i]) or np.isnan(rsi_4h_aligned[i]) or np.isnan(vol_spike_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vwap_val = vwap[i]
        rsi_val = rsi_4h_aligned[i]
        vol_spike = vol_spike_1d_aligned[i]
        
        if position == 0:
            # Enter long: price > VWAP, RSI > 50 (bullish), volume spike
            if price > vwap_val and rsi_val > 50 and vol_spike:
                signals[i] = 0.20
                position = 1
            # Enter short: price < VWAP, RSI < 50 (bearish), volume spike
            elif price < vwap_val and rsi_val < 50 and vol_spike:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price < VWAP or RSI <= 50 or no volume spike
            if price < vwap_val or rsi_val <= 50 or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price > VWAP or RSI >= 50 or no volume spike
            if price > vwap_val or rsi_val >= 50 or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals