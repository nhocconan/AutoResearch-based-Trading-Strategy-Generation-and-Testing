#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hEMA50_VolumeSpike_SessionFilter
Hypothesis: On 1h timeframe, use 4h EMA50 for trend direction and daily Camarilla R1/S1 levels for breakout entry with volume confirmation. Trade only during 08-20 UTC session to avoid low-liquidity periods. Discrete position sizing (0.20) minimizes fee churn. Designed for low trade frequency (15-35/year) to overcome 1h's historical difficulty (17% keep rate) by using higher timeframe structure for signal direction and 1h only for precise timing.
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
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla R1 and S1 from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].values
    
    # Camarilla R1 and S1 levels
    R1 = close_1d_prev + (high_1d - low_1d) * 1.1 / 12
    S1 = close_1d_prev - (high_1d - low_1d) * 1.1 / 12
    
    # Align Camarilla levels
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Volume confirmation: 1.8x average volume (balanced for 1h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stoploss (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of 4h EMA (50), volume MA (20), ATR (14)
    start_idx = max(50, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Skip if any data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(atr[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        ema_50_4h_val = ema_50_4h_aligned[i]
        R1_val = R1_aligned[i]
        S1_val = S1_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        
        if position == 0:
            # Look for breakouts in direction of 4h trend
            # Long: price breaks above R1 with volume confirmation and 4h uptrend
            long_signal = (high_val > R1_val) and (volume_val > 1.8 * vol_ma_val) and (close_val > ema_50_4h_val)
            # Short: price breaks below S1 with volume confirmation and 4h downtrend
            short_signal = (low_val < S1_val) and (volume_val > 1.8 * vol_ma_val) and (close_val < ema_50_4h_val)
            
            if long_signal:
                signals[i] = 0.20
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.20
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.20
            # Exit: ATR stoploss or trend reversal
            if (close_val < entry_price - 2.5 * atr_val or 
                close_val < ema_50_4h_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            # Exit: ATR stoploss or trend reversal
            if (close_val > entry_price + 2.5 * atr_val or 
                close_val > ema_50_4h_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hEMA50_VolumeSpike_SessionFilter"
timeframe = "1h"
leverage = 1.0