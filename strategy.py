#!/usr/bin/env python3
"""
6h_keltner_1w_trend_volume_v1
Hypothesis: On 6-hour timeframe, use weekly Keltner Channels with 1-week trend filter and volume confirmation.
Long when price closes above upper Keltner channel with weekly EMA(50) trending up and volume > 1.5x 20-period average.
Short when price closes below lower Keltner channel with weekly EMA(50) trending down and volume > 1.5x 20-period average.
Exit when price closes back inside the Keltner channel.
Designed for 10-30 trades/year to minimize fee dust while capturing strong trends with institutional validation.
Works in both bull/bear markets as Keltner channels adapt to volatility and weekly trend filter avoids counter-trend trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_keltner_1w_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Determine weekly trend direction (using EMA slope)
    weekly_trend_up = np.zeros(len(ema_50_1w_aligned), dtype=bool)
    weekly_trend_down = np.zeros(len(ema_50_1w_aligned), dtype=bool)
    for i in range(1, len(ema_50_1w_aligned)):
        if not np.isnan(ema_50_1w_aligned[i]) and not np.isnan(ema_50_1w_aligned[i-1]):
            weekly_trend_up[i] = ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1]
            weekly_trend_down[i] = ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1]
    
    # Calculate Keltner Channels on 6h timeframe
    # Typical Price = (High + Low + Close) / 3
    typical_price = (high + low + close) / 3
    atr_period = 10
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    ema_period = 20
    ema_tp = pd.Series(typical_price).ewm(span=ema_period, adjust=False, min_periods=ema_period).mean().values
    
    multiplier = 2.0
    upper_keltner = ema_tp + multiplier * atr
    lower_keltner = ema_tp - multiplier * atr
    
    # Volume filter: 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(max(20, 50), n):
        # Skip if data not available
        if (np.isnan(upper_keltner[i]) or np.isnan(lower_keltner[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
            
        # Volume confirmation
        vol_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price closes back inside Keltner channel (below upper)
            if close[i] < upper_keltner[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes back inside Keltner channel (above lower)
            if close[i] > lower_keltner[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only enter with volume confirmation and weekly trend alignment
            if vol_ok:
                # Long: price closes above upper Keltner with weekly uptrend
                if (close[i] > upper_keltner[i] and close[i-1] <= upper_keltner[i-1] and 
                    weekly_trend_up[i]):
                    position = 1
                    signals[i] = 0.25
                # Short: price closes below lower Keltner with weekly downtrend
                elif (close[i] < lower_keltner[i] and close[i-1] >= lower_keltner[i-1] and 
                      weekly_trend_down[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals