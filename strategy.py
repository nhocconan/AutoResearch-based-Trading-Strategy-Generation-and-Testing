#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_1wTrend_VolumeSpike
Hypothesis: Daily timeframe Donchian(20) breakout with weekly EMA50 trend filter and volume spike (>2.0x 20-period average). Long when price breaks above 20-day high in weekly uptrend with volume confirmation. Short when price breaks below 20-day low in weekly downtrend with volume confirmation. Exit via opposite Donchian level or ATR trailing stop (2.5*ATR from extreme). Designed for ~50-100 total trades over 4 years (12-25/year) via tight entry conditions on 1d timeframe.
"""

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
    
    # Get 1w data for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # need for EMA50 and Donchian
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian channels on 1d data (20-period)
    donch_period = 20
    donch_high = pd.Series(high).rolling(window=donch_period, min_periods=donch_period).max().values
    donch_low = pd.Series(low).rolling(window=donch_period, min_periods=donch_period).min().values
    
    # ATR for stoploss (20-period)
    atr_period = 20
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Volume regime: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_extreme = 0.0   # highest close since long entry
    short_extreme = 0.0  # lowest close since short entry
    
    # Start index: need warmup for calculations
    start_idx = max(donch_period, atr_period, 20, 50)  # +1 for roll in TR
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_trend = ema_50_1w_aligned[i]
        upper = donch_high[i]
        lower = donch_low[i]
        
        if position == 0:
            # Get 1w close aligned for trend comparison
            close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
            close_1w_val = close_1w_aligned[i]
            is_uptrend = close_1w_val > ema_trend
            
            if is_uptrend:  # 1w uptrend regime
                # Long: break above Donchian high with volume confirmation
                long_signal = (close[i] > upper) and vol_spike[i]
            else:  # 1w downtrend regime
                # Short: break below Donchian low with volume confirmation
                short_signal = (close[i] < lower) and vol_spike[i]
            
            if 'long_signal' in locals() and long_signal:
                signals[i] = 0.25
                position = 1
                long_extreme = close[i]
            elif 'short_signal' in locals() and short_signal:
                signals[i] = -0.25
                position = -1
                short_extreme = close[i]
            else:
                signals[i] = 0.0
                # Clear signal variables for next iteration
                if 'long_signal' in locals(): del long_signal
                if 'short_signal' in locals(): del short_signal
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Update extreme for trailing stop
            if close[i] > long_extreme:
                long_extreme = close[i]
            # Exit conditions: 
            # 1. ATR trailing stop (2.5*ATR from extreme)
            atr_stop = long_extreme - 2.5 * atr[i]
            # 2. Price breaks below Donchian low (opposite channel)
            if close[i] <= atr_stop or close[i] < lower:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Update extreme for trailing stop
            if close[i] < short_extreme:
                short_extreme = close[i]
            # Exit conditions:
            # 1. ATR trailing stop (2.5*ATR from extreme)
            atr_stop = short_extreme + 2.5 * atr[i]
            # 2. Price breaks above Donchian high (opposite channel)
            if close[i] >= atr_stop or close[i] > upper:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Donchian20_Breakout_1wTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0