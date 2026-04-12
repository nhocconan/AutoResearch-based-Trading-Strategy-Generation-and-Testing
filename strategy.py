#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend and Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h 20-period EMA (trend filter)
    close_4h = df_4h['close'].values
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # Calculate 4h 20-period high and low for Donchian channels
    high_20 = pd.Series(df_4h['high'].values).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_4h['low'].values).rolling(window=20, min_periods=20).min().values
    high_20_aligned = align_htf_to_ltf(prices, df_4h, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_4h, low_20)
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate 1d volume ratio: current volume / 10-period average
    vol_1d = df_1d['volume'].values
    vol_avg_10 = pd.Series(vol_1d).rolling(window=10, min_periods=10).mean().values
    vol_ratio_1d = vol_1d / np.where(vol_avg_10 > 0, vol_avg_10, np.nan)
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # Calculate 14-period ATR for volatility filter
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14 = np.full(n, np.nan)
    for i in range(13, n):
        atr14[i] = np.nanmean(tr[i-13:i+1])
    
    # Session filter: 8-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(ema20_4h_aligned[i]) or np.isnan(high_20_aligned[i]) or 
            np.isnan(low_20_aligned[i]) or np.isnan(atr14[i]) or 
            np.isnan(vol_ratio_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 8-20 UTC
        if not (8 <= hours[i] <= 20):
            if position == 1:
                signals[i] = 0.0
                position = 0
            elif position == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volatility filter: current ATR14 > 1.2x 20-period ATR EMA (elevated volatility)
        # Calculate 20-period ATR EMA on the fly
        atr_series = pd.Series(atr14[:i+1])
        atr_ema20_val = atr_series.ewm(span=20, adjust=False, min_periods=20).mean().iloc[-1] if len(atr_series) >= 20 else np.nan
        vol_filter = not np.isnan(atr_ema20_val) and atr14[i] > atr_ema20_val * 1.2
        
        # Trend filter: price above/below 4h 20 EMA
        price_above_ema20 = close[i] > ema20_4h_aligned[i]
        price_below_ema20 = close[i] < ema20_4h_aligned[i]
        
        # Entry conditions: Donchian breakout in direction of trend with volume confirmation
        long_breakout = close[i] > high_20_aligned[i]  # break above 4h 20-period high
        short_breakout = close[i] < low_20_aligned[i]  # break below 4h 20-period low
        volume_confirm = vol_ratio_1d_aligned[i] > 1.5  # volume > 1.5x 10-day average
        
        long_entry = long_breakout and price_above_ema20 and vol_filter and volume_confirm
        short_entry = short_breakout and price_below_ema20 and vol_filter and volume_confirm
        
        # Exit conditions: reversal signal or volatility contraction
        long_exit = (close[i] < ema20_4h_aligned[i]) or (atr14[i] < atr_ema20_val * 0.8 if not np.isnan(atr_ema20_val) else False)
        short_exit = (close[i] > ema20_4h_aligned[i]) or (atr14[i] < atr_ema20_val * 0.8 if not np.isnan(atr_ema20_val) else False)
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_4h_1d_donchian_ema20_vol_filter_v1"
timeframe = "1h"
leverage = 1.0