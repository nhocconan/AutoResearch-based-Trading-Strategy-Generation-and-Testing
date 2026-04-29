#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and ATR-based volatility filter
# Long when price breaks above Donchian(20) high AND price > 1d EMA50 AND ATR(14) < ATR(50) (low vol environment)
# Short when price breaks below Donchian(20) low AND price < 1d EMA50 AND ATR(14) < ATR(50)
# Exit when price retests Donchian midpoint OR trend filter fails
# Uses discrete position sizing (0.25) to minimize fee drag. Target: 19-50 trades/year on 4h timeframe.
# Donchian channels provide objective breakout levels, 1d EMA50 ensures trend alignment,
# ATR ratio filter avoids high-whipsaw environments. Works in bull via buying strength,
# in bear via selling weakness with trend filter preventing counter-trend traps.

name = "4h_Donchian20_1dEMA50_ATRFilter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels (20-period) on 4h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Calculate ATR(14) and ATR(50) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # First period has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_14 / atr_50  # Ratio < 1 indicates low volatility relative to longer term
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Donchian and EMA50 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(atr_ratio[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_ema50_1d = ema_50_1d_aligned[i]
        curr_donchian_high = donchian_high[i]
        curr_donchian_low = donchian_low[i]
        curr_donchian_mid = donchian_mid[i]
        curr_atr_ratio = atr_ratio[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price retests Donchian midpoint OR trend filter fails (price < 1d EMA50)
            if curr_close <= curr_donchian_mid or curr_close < curr_ema50_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price retests Donchian midpoint OR trend filter fails (price > 1d EMA50)
            if curr_close >= curr_donchian_mid or curr_close > curr_ema50_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Low volatility filter: ATR ratio < 1.0 (current vol < longer term avg vol)
            vol_filter = curr_atr_ratio < 1.0
            
            # Long when price breaks above Donchian high AND price > 1d EMA50 AND low vol
            if curr_close > curr_donchian_high and curr_close > curr_ema50_1d and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below Donchian low AND price < 1d EMA50 AND low vol
            elif curr_close < curr_donchian_low and curr_close < curr_ema50_1d and vol_filter:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals