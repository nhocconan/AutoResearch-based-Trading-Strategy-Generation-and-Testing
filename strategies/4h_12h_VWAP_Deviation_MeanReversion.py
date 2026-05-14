#!/usr/bin/env python3
"""
4h_12h_VWAP_Deviation_MeanReversion
Hypothesis: Price tends to revert to VWAP after significant deviations, especially when confirmed by higher timeframe trend (12h EMA50) and volume exhaustion. In bull markets, we buy dips to VWAP in uptrends; in bear markets, we sell rallies to VWAP in downtrends. This mean-reversion strategy works across regimes by aligning with the 12h trend. Uses tight entry conditions (2+ standard deviations from VWAP) to limit trades to ~30-50/year, reducing fee drag. Position size: 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data once for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema50_12h = np.zeros_like(close_12h)
    ema50_12h[0] = close_12h[0]
    alpha = 2.0 / (50 + 1)
    for i in range(1, len(close_12h)):
        ema50_12h[i] = alpha * close_12h[i] + (1 - alpha) * ema50_12h[i-1]
    
    # Align 12h EMA50 to 4h timeframe
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Main timeframe data (4h)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Typical price for VWAP calculation
    typical_price = (high + low + close) / 3.0
    
    # VWAP (20-period)
    vwap = np.full(n, np.nan)
    vol_sum = np.zeros(n)
    price_vol_sum = np.zeros(n)
    
    for i in range(n):
        vol_sum[i] = volume[i] + (vol_sum[i-1] if i >= 1 else 0)
        price_vol_sum[i] = typical_price[i] * volume[i] + (price_vol_sum[i-1] if i >= 1 else 0)
        if vol_sum[i] > 0:
            vwap[i] = price_vol_sum[i] / vol_sum[i]
        else:
            vwap[i] = typical_price[i]
    
    # Standard deviation of price from VWAP (20-period)
    price_dev = typical_price - vwap
    vwap_std = np.full(n, np.nan)
    
    for i in range(n):
        if i < 20:
            vwap_std[i] = np.std(price_dev[max(0, i-19):i+1]) if i >= 1 else 0.0
        else:
            vwap_std[i] = np.std(price_dev[i-20:i])
    
    # Avoid division by zero
    vwap_std = np.where(vwap_std == 0, 1e-10, vwap_std)
    
    # Z-score: how many standard deviations price is from VWAP
    z_score = price_dev / vwap_std
    
    # Volume filter: current volume < 0.5x 20-period average (volume exhaustion)
    volume_avg = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            volume_avg[i] = np.mean(volume[:i+1]) if i >= 0 else volume[i]
        else:
            volume_avg[i] = np.mean(volume[i-20:i])
    volume_filter = volume < (0.5 * volume_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(30, n):  # Start after VWAP warmup
        # Skip if NaN in critical values
        if np.isnan(vwap[i]) or np.isnan(vwap_std[i]) or np.isnan(ema50_12h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vwap_val = vwap[i]
        z = z_score[i]
        ema50 = ema50_12h_aligned[i]
        vol_exhausted = volume_filter[i]
        
        # Stoploss: 2.0 * ATR from entry (using 14-period ATR)
        # Calculate ATR on the fly for simplicity in stop condition
        if i >= 14:
            tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            # Simplified ATR: use recent true range average
            tr_sum = 0
            for j in range(1, 15):
                if i - j >= 0:
                    tr_sum += max(high[i-j] - low[i-j], abs(high[i-j] - close[i-j-1]), abs(low[i-j] - close[i-j-1]))
            atr_est = tr_sum / 14
        else:
            atr_est = 0
        
        if position == 1 and price < entry_price - 2.0 * atr_est:
            signals[i] = 0.0
            position = 0
            continue
        elif position == -1 and price > entry_price + 2.0 * atr_est:
            signals[i] = 0.0
            position = 0
            continue
        
        if position == 0:
            # Long: price significantly below VWAP (oversold) in uptrend with volume exhaustion
            if z < -2.0 and price > ema50 and vol_exhausted:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price significantly above VWAP (overbought) in downtrend with volume exhaustion
            elif z > 2.0 and price < ema50 and vol_exhausted:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price returns to VWAP or trend breaks
            if z > -0.5 or price < ema50:  # Return to VWAP or trend breakdown
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to VWAP or trend breaks
            if z < 0.5 or price > ema50:  # Return to VWAP or trend breakdown
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_12h_VWAP_Deviation_MeanReversion"
timeframe = "4h"
leverage = 1.0