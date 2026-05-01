#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume spike confirmation.
# Long when price breaks above 20-bar 12h Donchian upper band AND price > 1d EMA50 AND volume > 1.5x 20-bar 12h volume MA.
# Short when price breaks below 20-bar 12h Donchian lower band AND price < 1d EMA50 AND volume > 1.5x 20-bar 12h volume MA.
# Uses discrete sizing 0.25 to minimize fee churn. Designed to capture medium-term trends in both bull and bear markets.

name = "12h_Donchian20_1dEMA50_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 12h data ONCE before loop for Donchian channels and volume MA
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h Donchian(20) channels
    donch_hi = pd.Series(df_12h['high'].values).rolling(window=20, min_periods=20).max().values
    donch_lo = pd.Series(df_12h['low'].values).rolling(window=20, min_periods=20).min().values
    donch_hi_aligned = align_htf_to_ltf(prices, df_12h, donch_hi)
    donch_lo_aligned = align_htf_to_ltf(prices, df_12h, donch_lo)
    
    # 12h volume MA(20)
    vol_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA50 and Donchian
    
    for i in range(start_idx, n):
        # Session filter: 00-23 UTC (12h timeframe, trade all sessions)
        hour = hours[i]
        in_session = True  # 12h bars cover full day, no session filter needed
        
        # Skip if any data not ready
        if (np.isnan(donch_hi_aligned[i]) or np.isnan(donch_lo_aligned[i]) or 
            np.isnan(vol_ma_12h_aligned[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_donch_hi = donch_hi_aligned[i]
        curr_donch_lo = donch_lo_aligned[i]
        curr_vol_ma = vol_ma_12h_aligned[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        
        # Volume confirmation: current 12h volume > 1.5x 20-period average
        volume_confirm = curr_vol > (curr_vol_ma * 1.5)
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: price > Donchian upper AND price > 1d EMA50 AND volume confirmation
            if (curr_close > curr_donch_hi and 
                curr_close > curr_ema_50_1d and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: price < Donchian lower AND price < 1d EMA50 AND volume confirmation
            elif (curr_close < curr_donch_lo and 
                  curr_close < curr_ema_50_1d and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price < Donchian lower (breakdown) OR price < 1d EMA50 (trend violation)
            if (curr_close < curr_donch_lo or 
                curr_close < curr_ema_50_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price > Donchian upper (breakout) OR price > 1d EMA50 (trend violation)
            if (curr_close > curr_donch_hi or 
                curr_close > curr_ema_50_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals