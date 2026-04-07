#!/usr/bin/env python3
"""
6h_adaptive_volume_profile_v1
Hypothesis: On 6h timeframe, use volume profile to identify high-volume nodes (HVN) as support/resistance.
In uptrend (price > 1w EMA200): long when price pulls back to HVN with volume confirmation.
In downtrend (price < 1w EMA200): short when price rallies to HVN with volume confirmation.
Uses volume-weighted average price (VWAP) over 20 periods to define fair value, and trades mean reversion
to HVN levels identified from 1d volume profile. Designed to work in both bull (buy dips) and bear (sell rallies)
markets by adapting to the higher timeframe trend.
Target: 12-37 trades/year (~50-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_adaptive_volume_profile_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = volumes = prices['volume'].values
    
    # 1d data for volume profile
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1w EMA200 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    ema_200 = df_1w['close'].ewm(span=200, adjust=False).mean()
    ema_200_aligned = align_htf_to_ltf(prices, df_1w, ema_200.values)
    
    # Calculate volume profile from 1d data (previous 20 days)
    # We'll use the volume-weighted average price as our fair value reference
    # and identify high volume nodes
    def calculate_vwap(high, low, close, volume, window):
        """Calculate VWAP over a window"""
        typical_price = (high + low + close) / 3.0
        vwap_num = np.convolve(typical_price * volume, np.ones(window), 'same')
        vwap_den = np.convolve(volume, np.ones(window), 'same')
        # Avoid division by zero
        vwap_den = np.where(vwap_den == 0, 1, vwap_den)
        return vwap_num / vwap_den
    
    # Calculate 20-period VWAP on 1d data
    vwap_20d = calculate_vwap(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values,
        df_1d['volume'].values,
        20
    )
    
    # Identify high volume nodes: days where volume > 1.5x 20-day average volume
    vol_ma_20d = df_1d['volume'].rolling(window=20, min_periods=20).mean().values
    high_volume_days = df_1d['volume'].values > 1.5 * vol_ma_20d
    
    # For each high volume day, the typical price is a potential HVN
    hvn_prices = np.where(
        high_volume_days,
        (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3.0,
        np.nan
    )
    
    # We'll use the most recent significant HVN as our reference
    # For simplicity, we'll use the VWAP itself as dynamic fair value
    # and look for deviations from it
    vwap_20d_aligned = align_htf_to_ltf(prices, df_1d, vwap_20d)
    
    # 6-period VWAP on 6h data for entry timing
    typical_price_6h = (high + low + close) / 3.0
    vwap_num = np.convolve(typical_price_6h * volume, np.ones(6), 'same')
    vwap_den = np.convolve(volume, np.ones(6), 'same')
    vwap_den = np.where(vwap_den == 0, 1, vwap_den)
    vwap_6h = vwap_num / vwap_den
    
    # Standard deviation of price from VWAP for volatility normalization
    price_dev = typical_price_6h - vwap_6h
    # Use 20-period rolling std dev of price deviation
    price_dev_series = pd.Series(price_dev)
    dev_std = price_dev_series.rolling(window=20, min_periods=20).std().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema_200_aligned[i]) or np.isnan(vwap_20d_aligned[i]) or
            np.isnan(vwap_6h[i]) or np.isnan(dev_std[i]) or dev_std[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x average volume
        vol_ma_6h = np.convolve(volume, np.ones(6), 'same')[i] if i >= 5 else volume[i]
        vol_ma_6h = max(vol_ma_6h, 1e-10)  # Avoid division by zero
        vol_confirm = volume[i] > 1.3 * vol_ma_6h
        
        # Calculate z-score of price deviation from VWAP
        if dev_std[i] > 0:
            z_score = price_dev[i] / dev_std[i]
        else:
            z_score = 0
        
        if position == 1:  # Long position
            # Exit: price returns to VWAP (mean reversion) or trend turns bearish
            if z_score >= -0.1 or close[i] < ema_200_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price returns to VWAP (mean reversion) or trend turns bullish
            if z_score <= 0.1 or close[i] > ema_200_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only trade in direction of higher timeframe trend
            if close[i] > ema_200_aligned[i]:  # Uptrend
                # Long when price deviates significantly below VWAP (oversold)
                if z_score <= -1.5 and vol_confirm:
                    position = 1
                    signals[i] = 0.25
            else:  # Downtrend
                # Short when price deviates significantly above VWAP (overbought)
                if z_score >= 1.5 and vol_confirm:
                    position = -1
                    signals[i] = -0.25
    
    return signals