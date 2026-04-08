#!/usr/bin/env python3
# 4h_vwap_reversion_1d_trend_volume_v1
# Hypothesis: Mean reversion to VWAP works in ranging markets, while trend filter prevents losses in trends.
# Long when price closes below VWAP(20) with volume > 1.5x avg and 1d uptrend.
# Short when price closes above VWAP(20) with volume > 1.5x avg and 1d downtrend.
# Exit when price returns to VWAP(20) or opposite signal occurs.
# Uses VWAP for mean reversion and 1d EMA50 for trend filter.
# Target: 20-40 trades/year to avoid excessive fee drag while capturing mean reversion moves.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_vwap_reversion_1d_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d trend filter: EMA50
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    daily_close = df_daily['close'].values
    daily_ema50 = pd.Series(daily_close).ewm(span=50, min_periods=50, adjust=False).mean().values
    daily_ema50_4h = align_htf_to_ltf(prices, df_daily, daily_ema50)
    
    # Calculate VWAP(20) - Volume Weighted Average Price
    typical_price = (high + low + close) / 3.0
    vwap_sum = pd.Series(typical_price * volume).rolling(window=20, min_periods=20).sum().values
    volume_sum = pd.Series(volume).rolling(window=20, min_periods=20).sum().values
    vwap = np.where(volume_sum != 0, vwap_sum / volume_sum, 0.0)
    
    # Volume confirmation: 20-period average on 4h
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(daily_ema50_4h[i]) or np.isnan(vwap[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to VWAP or opposite signal
            if close[i] >= vwap[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to VWAP or opposite signal
            if close[i] <= vwap[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Daily trend filter
            daily_uptrend = close[i] > daily_ema50_4h[i]
            daily_downtrend = close[i] < daily_ema50_4h[i]
            
            # Long entry: price closes below VWAP with volume and uptrend
            if close[i] < vwap[i] and volume_ok and daily_uptrend:
                position = 1
                signals[i] = 0.25
            # Short entry: price closes above VWAP with volume and downtrend
            elif close[i] > vwap[i] and volume_ok and daily_downtrend:
                position = -1
                signals[i] = -0.25
    
    return signals