#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot (R1/S1) breakout with 1d trend filter and volume confirmation
# Camarilla levels provide high-probability reversal/breakout levels from prior day's range
# 1d EMA50 filters for daily trend alignment to avoid counter-trend trades
# Volume > 1.5x 20-period average confirms institutional participation
# Designed for 12h timeframe with selective entries to avoid overtrading
# Target: 12-37 trades per year per symbol (50-150 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 50-period EMA on 1d timeframe for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Camarilla levels (R1, S1) from prior day's range
    # R1 = close + (high - low) * 1.12
    # S1 = close - (high - low) * 1.12
    camarilla_high = close_1d + (high_1d - low_1d) * 1.12
    camarilla_low = close_1d - (high_1d - low_1d) * 1.12
    camarilla_high_aligned = align_htf_to_ltf(prices, df_1d, camarilla_high)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_1d, camarilla_low)
    
    # Calculate ATR for stop loss
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter: volume > 1.5x 20-period average
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if NaN in indicators
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(camarilla_high_aligned[i]) or \
           np.isnan(camarilla_low_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 1d trend
        is_uptrend = close[i] > ema50_1d_aligned[i]
        is_downtrend = close[i] < ema50_1d_aligned[i]
        
        # Volume confirmation
        has_volume = vol_filter[i]
        
        price = close[i]
        
        if position == 0:
            # Long entry: price breaks above Camarilla R1 + uptrend + volume
            long_signal = (price > camarilla_high_aligned[i]) and is_uptrend and has_volume
            
            # Short entry: price breaks below Camarilla S1 + downtrend + volume
            short_signal = (price < camarilla_low_aligned[i]) and is_downtrend and has_volume
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: stop loss or Camarilla S1 break
            stop_loss = entry_price - 2.0 * atr[i]
            camarilla_break = price < camarilla_low_aligned[i]
            
            if stop_loss <= 0 or price <= stop_loss or camarilla_break:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: stop loss or Camarilla R1 break
            stop_loss = entry_price + 2.0 * atr[i]
            camarilla_break = price > camarilla_high_aligned[i]
            
            if stop_loss <= 0 or price >= stop_loss or camarilla_break:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_1dTrendFilter_Volume"
timeframe = "12h"
leverage = 1.0