# 12h Institutional Flow Strategy with 1d Trend and Volume Confirmation
# Hypothesis: Institutional money flows often manifest as sustained price moves accompanied by volume.
# On 12h timeframe, we combine: (1) price breaking above/below 1d VWAP bands (institutional interest),
# (2) volume > 2x 4-period average (institutional participation), and (3) 1d EMA50 trend filter.
# This captures sustained moves while avoiding chop. Works in bull/bear by aligning with 1d trend.
# Target: 15-35 trades/year to minimize fee drag.

from typing import Any
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
    
    # Get 1d data for VWAP and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d VWAP (volume-weighted average price)
    vwap_1d = np.full(len(df_1d), np.nan)
    for i in range(len(df_1d)):
        typical_price = (high_1d[i] + low_1d[i] + close_1d[i]) / 3.0
        vwap_1d[i] = np.sum(volume_1d[:i+1] * typical_price) / np.sum(volume_1d[:i+1]) if np.sum(volume_1d[:i+1]) > 0 else typical_price
    
    # Calculate 1d VWAP upper/lower bands (1 standard deviation)
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    vwap_dev = np.full(len(df_1d), np.nan)
    for i in range(len(df_1d)):
        if i >= 19:  # 20-period lookback for std dev
            vwap_dev[i] = np.std(typical_price_1d[i-19:i+1])
        else:
            vwap_dev[i] = np.std(typical_price_1d[:i+1]) if i > 0 else 0.0
    
    vwap_upper_1d = vwap_1d + vwap_dev
    vwap_lower_1d = vwap_1d - vwap_dev
    
    # Align VWAP bands to 12h timeframe
    vwap_upper_aligned = align_htf_to_ltf(prices, df_1d, vwap_upper_1d)
    vwap_lower_aligned = align_htf_to_ltf(prices, df_1d, vwap_lower_1d)
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # 1d EMA50 trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: volume > 2.0 x 4-period average (2 days of 12h bars)
    vol_ma_4 = np.full(n, np.nan)
    for i in range(3, n):
        vol_ma_4[i] = np.mean(volume[i-3:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need 1d VWAP (20), EMA (50), volume MA (4)
    start_idx = max(20, 50, 4)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(vwap_upper_aligned[i]) or np.isnan(vwap_lower_aligned[i]) or 
            np.isnan(vwap_aligned[i]) or np.isnan(ema_50_aligned[i]) or
            np.isnan(vol_ma_4[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_4[i]
        
        # Volume filter: significant volume spike
        vol_filter = vol_now > 2.0 * vol_avg
        
        # Trend filter from 1d EMA50
        bullish_trend = price > ema_50_aligned[i]
        bearish_trend = price < ema_50_aligned[i]
        
        if position == 0:
            # Long: price above VWAP upper band with volume and bullish trend
            if price > vwap_upper_aligned[i] and vol_filter and bullish_trend:
                signals[i] = size
                position = 1
            # Short: price below VWAP lower band with volume and bearish trend
            elif price < vwap_lower_aligned[i] and vol_filter and bearish_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to VWAP (mean reversion) or trend turns bearish
            if price <= vwap_aligned[i] or not bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to VWAP (mean reversion) or trend turns bullish
            if price >= vwap_aligned[i] or not bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Institutional_Flow_VWAP_Bands_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0