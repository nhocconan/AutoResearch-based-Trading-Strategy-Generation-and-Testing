#!/usr/bin/env python3
"""
4h_VWAP_Reversion_1dTrend_VolumeFilter
Hypothesis: Mean reversion to VWAP on 4h with 1d EMA50 trend filter and volume spike confirmation.
Long when price < VWAP in uptrend with volume spike, short when price > VWAP in downtrend with volume spike.
Exit when price crosses VWAP or trend reverses. Designed for low trade frequency and robustness in both bull and bear markets.
"""

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
    
    # Get 4h data for VWAP calculation (primary timeframe)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Typical price for VWAP
    typical_price_4h = (high_4h + low_4h + close_4h) / 3.0
    
    # VWAP: cumulative(typical_price * volume) / cumulative(volume) reset each 4h bar
    # We'll use session VWAP (within each 4h bar) approximated by rolling 20-period
    pv_4h = typical_price_4h * volume_4h
    vol_sum_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).sum().values
    pv_sum_4h = pd.Series(pv_4h).rolling(window=20, min_periods=20).sum().values
    vwap_4h = pv_sum_4h / vol_sum_4h
    vwap_4h = np.where(vol_sum_4h == 0, typical_price_4h, vwap_4h)  # avoid div by zero
    
    # Align VWAP to original timeframe
    vwap_aligned = align_htf_to_ltf(prices, df_4h, vwap_4h)
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Daily EMA50 for trend direction
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: volume > 2.0x 20-period average on LTF
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vwap_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_trend = ema_50_1d_aligned[i]
        vwap = vwap_aligned[i]
        
        if position == 0:
            # Regime-based entry logic
            if close[i] > ema_trend:  # Uptrend regime (daily)
                # Long: mean reversion to VWAP (price below VWAP) with volume spike
                long_signal = (close[i] < vwap) and vol_spike[i]
                # Short: only on extreme deviation (mean reversion fade)
                short_signal = (close[i] > vwap * 1.02) and vol_spike[i] and (volume[i] > (4.0 * vol_ma_20[i]))
            else:  # Downtrend regime (daily)
                # Short: mean reversion to VWAP (price above VWAP) with volume spike
                short_signal = (close[i] > vwap) and vol_spike[i]
                # Long: only on extreme deviation (mean reversion fade)
                long_signal = (close[i] < vwap * 0.98) and vol_spike[i] and (volume[i] > (4.0 * vol_ma_20[i]))
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions: price crosses above VWAP or trend reversal
            exit_signal = (close[i] > vwap) or (close[i] < ema_trend * 0.99)
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions: price crosses below VWAP or trend reversal
            exit_signal = (close[i] < vwap) or (close[i] > ema_trend * 1.01)
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_VWAP_Reversion_1dTrend_VolumeFilter"
timeframe = "4h"
leverage = 1.0