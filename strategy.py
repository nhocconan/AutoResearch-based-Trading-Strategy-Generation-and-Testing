# 6h_MultiTimeframeConfluence_v1
# 6h timeframe with 1d and 1w trend filters
# Uses confluence of: 6h price above/below 1d VWAP, 1d candle direction, and 1w EMA trend
# Designed to work in both bull and bear markets by requiring multi-timeframe agreement
# Target: 15-25 trades/year per symbol (60-100 total over 4 years) to minimize fee drag

#!/usr/bin/env python3
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
    
    # Load 1d data ONCE for VWAP and candle direction
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d VWAP: cumulative (price * volume) / cumulative volume
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    pv_1d = typical_price_1d * volume_1d
    cum_pv = np.nancumsum(pv_1d)
    cum_vol = np.nancumsum(volume_1d)
    vwap_1d = np.divide(cum_pv, cum_vol, out=np.full_like(cum_pv, np.nan), where=cum_vol!=0)
    
    # 1d candle direction: 1 for bullish (close > open), -1 for bearish
    open_1d = df_1d['open'].values
    candle_dir_1d = np.where(close_1d > open_1d, 1, -1)
    
    # Load 1w data ONCE for EMA trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 20-period EMA on 1w
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align indicators to 6h timeframe
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    candle_dir_1d_aligned = align_htf_to_ltf(prices, df_1d, candle_dir_1d)
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 20  # Need EMA warmup
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(vwap_1d_aligned[i]) or 
            np.isnan(candle_dir_1d_aligned[i]) or
            np.isnan(ema_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Multi-timeframe confluence conditions
        price_above_vwap = close[i] > vwap_1d_aligned[i]
        price_below_vwap = close[i] < vwap_1d_aligned[i]
        bullish_1d = candle_dir_1d_aligned[i] == 1
        bearish_1d = candle_dir_1d_aligned[i] == -1
        price_above_1w_ema = close[i] > ema_20_1w_aligned[i]
        price_below_1w_ema = close[i] < ema_20_1w_aligned[i]
        
        if position == 0:
            # Long: price above 1d VWAP + bullish 1d candle + price above 1w EMA
            if (price_above_vwap and 
                bullish_1d and 
                price_above_1w_ema):
                position = 1
                signals[i] = position_size
            # Short: price below 1d VWAP + bearish 1d candle + price below 1w EMA
            elif (price_below_vwap and 
                  bearish_1d and 
                  price_below_1w_ema):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: any condition fails
            if not (price_above_vwap and bullish_1d and price_above_1w_ema):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: any condition fails
            if not (price_below_vwap and bearish_1d and price_below_1w_ema):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_MultiTimeframeConfluence_v1"
timeframe = "6h"
leverage = 1.0