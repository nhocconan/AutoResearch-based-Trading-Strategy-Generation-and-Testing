#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h/1d for trend direction and 1h for entry timing.
# Trend direction: 4h close above/below 4h EMA200 (bull/bear).
# Entry: In bull trend, buy when 1h price crosses above 1h VWAP; in bear trend, sell when 1h price crosses below 1h VWAP.
# Volume filter: Require volume > 1.2x 20-period average to confirm breakouts.
# Session filter: Trade only between 08:00-20:00 UTC to avoid low-volume Asian session.
# Position size: 0.20 (20%) to manage drawdown in volatile markets.
# This approach uses higher timeframes for trend filtering (reducing whipsaw) and lower timeframe for precise entries,
# while volume and session filters reduce false signals. Designed to work in both bull and bear markets by
# following the trend defined on 4h and using mean-reversion to VWAP on 1h for entries.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter (EMA200)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 200:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    # Calculate EMA200 on 4h close
    ema_200_4h = np.full(len(close_4h), np.nan)
    if len(close_4h) >= 200:
        ema_200_4h[199] = np.mean(close_4h[:200])  # Simple average for first value
        for i in range(200, len(close_4h)):
            ema_200_4h[i] = (close_4h[i] * 2 / (200 + 1)) + (ema_200_4h[i-1] * (199 / (200 + 1)))
    
    # Align 4h EMA200 to 1h timeframe
    ema_200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_200_4h)
    
    # Calculate 1h VWAP (typical price * volume cumulative)
    typical_price = (high + low + close) / 3.0
    vwap_num = np.cumsum(typical_price * volume)
    vwap_den = np.cumsum(volume)
    vwap = np.divide(vwap_num, vwap_den, out=np.full_like(vwap_num, np.nan), where=vwap_den!=0)
    
    # Average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # Precompute session hours (08:00-20:00 UTC)
    hours = pd.to_datetime(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.20  # 20% position size
    
    for i in range(200, n):  # Start after EMA200 warmup
        # Skip if any required data is not ready
        if (np.isnan(ema_200_4h_aligned[i]) or np.isnan(vwap[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08:00-20:00 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_200 = ema_200_4h_aligned[i]
        vwap_val = vwap[i]
        
        # Volume confirmation: current volume > 1.2x average volume
        volume_confirm = vol > 1.2 * avg_vol
        
        if position == 0:
            # Determine trend from 4h EMA200
            if price > ema_200:  # Bull trend
                # Long: price crosses above VWAP + volume confirmation
                if price > vwap_val and volume_confirm:
                    position = 1
                    signals[i] = position_size
            else:  # Bear trend (price <= ema_200)
                # Short: price crosses below VWAP + volume confirmation
                if price < vwap_val and volume_confirm:
                    position = -1
                    signals[i] = -position_size
        elif position == 1:
            # Exit long: price crosses below VWAP (mean reversion)
            if price < vwap_val:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above VWAP (mean reversion)
            if price > vwap_val:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_4h_VWAP_Trend_Filter"
timeframe = "1h"
leverage = 1.0