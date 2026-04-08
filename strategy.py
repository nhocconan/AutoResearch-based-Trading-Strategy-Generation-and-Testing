#!/usr/bin/env python3
# 1h_momentum_confluence_volume
# Hypothesis: On 1h timeframe, use 4h RSI momentum with 1d volume confirmation to capture momentum bursts.
# Long when 4h RSI crosses above 60 with 1h volume > 1.5x average and price above 1h VWAP.
# Short when 4h RSI crosses below 40 with 1h volume > 1.5x average and price below 1h VWAP.
# Exit when RSI returns to 50 (neutral zone).
# Uses RSI thresholds (60/40) to avoid whipsaws, volume confirmation to filter noise, VWAP for intraday trend.
# Target: 15-37 trades/year (~60-150 total over 4 years) to stay within fee limits.
# Works in bull/bear by capturing momentum shifts regardless of direction.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_momentum_confluence_volume"
timeframe = "1h"
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
    
    # Calculate 4h RSI (14-period)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[14] = np.mean(gain[1:15])
    avg_loss[14] = np.mean(loss[1:15])
    
    for i in range(15, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi_4h = np.where(avg_loss == 0, 100, 100 - (100 / (1 + rs)))
    
    # Align 4h RSI to 1h
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # Calculate 1h VWAP (typical price * volume / cumulative volume)
    typical_price = (high + low + close) / 3
    vwap_numerator = np.cumsum(typical_price * volume)
    vwap_denominator = np.cumsum(volume)
    vwap = np.where(vwap_denominator > 0, vwap_numerator / vwap_denominator, typical_price)
    
    # Volume confirmation: 20-period average on 1h
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(rsi_4h_aligned[i]) or np.isnan(vwap[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI returns to 50 (neutral)
            if rsi_4h_aligned[i] <= 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: RSI returns to 50 (neutral)
            if rsi_4h_aligned[i] >= 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # RSI momentum conditions
            rsi_bullish = rsi_4h_aligned[i] > 60 and (i == start_idx or rsi_4h_aligned[i-1] <= 60)
            rsi_bearish = rsi_4h_aligned[i] < 40 and (i == start_idx or rsi_4h_aligned[i-1] >= 40)
            
            # Price relative to VWAP for intraday trend
            price_above_vwap = close[i] > vwap[i]
            price_below_vwap = close[i] < vwap[i]
            
            # Long entry: RSI crosses above 60 with volume and price above VWAP
            if rsi_bullish and volume_ok and price_above_vwap:
                position = 1
                signals[i] = 0.20
            # Short entry: RSI crosses below 40 with volume and price below VWAP
            elif rsi_bearish and volume_ok and price_below_vwap:
                position = -1
                signals[i] = -0.20
    
    return signals