#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_vwap_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for VWAP
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return signals
    
    # Pre-compute 1d VWAP
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    vwap = (typical_price * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    vwap_array = vwap.values
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_array)
    
    # Pre-compute 4h VWAP
    typical_price_4h = (high + low + close) / 3
    vwap_4h = (typical_price_4h * volume).cumsum() / volume.cumsum()
    
    # Pre-compute 4h VWAP slope (rate of change over 5 periods)
    vwap_series = pd.Series(vwap_4h)
    vwap_slope = vwap_series.diff(periods=5) / 5
    vwap_slope_array = vwap_slope.values
    
    # Pre-compute 4h volume ratio (current volume / 20-period average)
    volume_series = pd.Series(volume)
    volume_sma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(vwap_1d_aligned[i]) or 
            np.isnan(vwap_slope_array[i]) or 
            np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        price_vwap_4h = vwap_4h[i]
        vwap_slope_current = vwap_slope_array[i]
        volume_current = volume[i]
        
        # VWAP trend: slope positive = bullish, negative = bearish
        vwap_bullish = vwap_slope_current > 0
        vwap_bearish = vwap_slope_current < 0
        
        # Volume confirmation: current volume > 1.3x 20-period average
        vol_confirm = volume_current > 1.3 * volume_sma_20[i]
        
        # Price relative to 1d VWAP
        price_above_1d_vwap = price_close > vwap_1d_aligned[i]
        price_below_1d_vwap = price_close < vwap_1d_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: VWAP trending up + price above 1d VWAP + volume confirmation
        if vwap_bullish and price_above_1d_vwap and vol_confirm:
            enter_long = True
        
        # Short: VWAP trending down + price below 1d VWAP + volume confirmation
        if vwap_bearish and price_below_1d_vwap and vol_confirm:
            enter_short = True
        
        # Exit conditions: opposite VWAP trend or price crosses 1d VWAP
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if VWAP trend turns bearish OR price crosses below 1d VWAP
            exit_long = vwap_bearish or (not price_above_1d_vwap)
        elif position == -1:
            # Exit short if VWAP trend turns bullish OR price crosses above 1d VWAP
            exit_short = vwap_bullish or (not price_below_1d_vwap)
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals