# 6h_1d_1w_VolumeWeighted_CCI_Trend_Filtered_v1
# Hypothesis: CCI(20) on 6h combined with 1d trend filter and 1d volume-weighted CCI confirmation.
# Long when: 6h CCI crosses above -100, 1d EMA20 > EMA50, and 1d volume-weighted CCI > -50.
# Short when: 6h CCI crosses below +100, 1d EMA20 < EMA50, and 1d volume-weighted CCI < 50.
# Exit when 6h CCI crosses zero in opposite direction or trend fails.
# Uses volume-weighted price for more institutional-grade signals.
# Target: 20-40 trades/year per symbol.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data once for trend and volume-weighted CCI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d EMA20 and EMA50 for trend filter
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Typical price for CCI calculation
    tp_1d = (high_1d + low_1d + close_1d) / 3.0
    
    # Volume-weighted typical price (VWTP)
    vwtp_1d = (tp_1d * volume_1d) / (volume_1d + 1e-10)  # Avoid division by zero
    
    # CCI calculation on 1d using VWTP
    # CCI = (VWTP - SMA) / (0.015 * Mean Deviation)
    sma_vwtp = pd.Series(vwtp_1d).rolling(window=20, min_periods=20).mean()
    mad_vwtp = pd.Series(vwtp_1d).rolling(window=20, min_periods=20).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    vwtp_cci_1d = (vwtp_1d - sma_vwtp.values) / (0.015 * mad_vwtp.values + 1e-10)
    vwtp_cci_1d = vwtp_cci_1d.values
    
    # Align 1d indicators to 6h
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    vwtp_cci_1d_aligned = align_htf_to_ltf(prices, df_1d, vwtp_cci_1d)
    
    # Load 6h data for CCI calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Typical price for 6h CCI
    tp_6h = (high_6h + low_6h + close_6h) / 3.0
    
    # CCI calculation on 6h
    sma_tp = pd.Series(tp_6h).rolling(window=20, min_periods=20).mean()
    mad_tp = pd.Series(tp_6h).rolling(window=20, min_periods=20).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    cci_6h = (tp_6h - sma_tp.values) / (0.015 * mad_tp.values + 1e-10)
    cci_6h = cci_6h.values
    
    # Align 6h CCI to its own timeframe (no alignment needed as it's already 6h)
    # But we need to align it to the main price timeframe (which is also 6h)
    # Since main price is 6h, we can use it directly
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(ema20_1d_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vwtp_cci_1d_aligned[i]) or np.isnan(cci_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        
        # Trend condition: 1d EMA20 > EMA50 for long, < for short
        uptrend = ema20_1d_aligned[i] > ema50_1d_aligned[i]
        downtrend = ema20_1d_aligned[i] < ema50_1d_aligned[i]
        
        # Volume-weighted CCI condition
        vwtp_cci = vwtp_cci_1d_aligned[i]
        vwtp_cci_bullish = vwtp_cci > -50
        vwtp_cci_bearish = vwtp_cci < 50
        
        # 6h CCI signals
        cci_now = cci_6h[i]
        cci_prev = cci_6h[i-1] if i > 0 else cci_6h[i]
        
        # Bullish crossover: CCI crosses above -100
        bullish_cross = (cci_prev <= -100) and (cci_now > -100)
        # Bearish crossover: CCI crosses below +100
        bearish_cross = (cci_prev >= 100) and (cci_now < 100)
        # Exit signals: CCI crosses zero
        exit_long = (cci_prev > 0) and (cci_now <= 0)
        exit_short = (cci_prev < 0) and (cci_now >= 0)
        
        if position == 0:
            # Long conditions
            if bullish_cross and uptrend and vwtp_cci_bullish:
                signals[i] = 0.25
                position = 1
            # Short conditions
            elif bearish_cross and downtrend and vwtp_cci_bearish:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: CCI crosses zero down OR trend fails OR VWTP CCI too bearish
            if exit_long or not uptrend or vwtp_cci < -100:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: CCI crosses zero up OR trend fails OR VWTP CCI too bullish
            if exit_short or not downtrend or vwtp_cci > 100:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1d_1w_VolumeWeighted_CCI_Trend_Filtered_v1"
timeframe = "6h"
leverage = 1.0