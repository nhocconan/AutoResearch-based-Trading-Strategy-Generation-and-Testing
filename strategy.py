#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data for HTF reference
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 12-hour high/low for Donchian channel (20-period)
    # Since we're using 12h timeframe, we need to aggregate 1d data to get 12h bars
    # We'll use the daily data and calculate rolling window on 12h equivalent
    # For 12h timeframe, we'll use 2-period lookback on daily data (since 2*12h = 24h = 1d)
    # But to be more precise, we'll calculate 12h high/low from intraday data if available
    # Since we only have daily data in HTF, we'll approximate 12h channels using daily
    
    # For 12h timeframe, we'll calculate Donchian channels using 10-period (equivalent to 5 days)
    # But since we're working with daily data, we'll use the daily high/low directly
    
    # Calculate 20-period Donchian channels from daily data
    # Highest high over last 20 days
    highest_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Lowest low over last 20 days
    lowest_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align to 12h timeframe
    highest_high_aligned = align_htf_to_ltf(prices, df_1d, highest_high)
    lowest_low_aligned = align_htf_to_ltf(prices, df_1d, lowest_low)
    
    # Calculate 12h close price (we'll use close price from prices dataframe)
    close_price = prices['close'].values
    
    # Volume confirmation - use 12h volume from prices
    volume_12h = prices['volume'].values
    vol_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # ADX calculation for trend strength (using daily data)
    # Calculate True Range
    high_low = high_1d - low_1d
    high_close = np.abs(high_1d - np.roll(close_1d, 1))
    low_close = np.abs(low_1d - np.roll(close_1d, 1))
    high_low[0] = high_1d[0] - low_1d[0]
    high_close[0] = np.abs(high_1d[0] - close_1d[0])
    low_close[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    tr[0] = high_low[0]
    
    # Calculate Directional Movement
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    # Calculate smoothed values
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_dm_14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm_14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    # Avoid division by zero
    plus_di_14 = np.zeros_like(tr_14)
    minus_di_14 = np.zeros_like(tr_14)
    mask = tr_14 != 0
    plus_di_14[mask] = 100 * plus_dm_14[mask] / tr_14[mask]
    minus_di_14[mask] = 100 * minus_dm_14[mask] / tr_14[mask]
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14 + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if (np.isnan(highest_high_aligned[i]) or np.isnan(lowest_low_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_price[i]
        vol = volume_12h[i]
        
        if position == 0:
            # Long: price breaks above Donchian high, strong trend (ADX > 25), volume confirmation
            if (price > highest_high_aligned[i] and 
                adx_aligned[i] > 25 and 
                vol > 1.5 * vol_ma_12h[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low, strong trend (ADX > 25), volume confirmation
            elif (price < lowest_low_aligned[i] and 
                  adx_aligned[i] > 25 and 
                  vol > 1.5 * vol_ma_12h[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below Donchian low or trend weakens (ADX < 20)
            if price < lowest_low_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above Donchian high or trend weakens (ADX < 20)
            if price > highest_high_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_ADX25_VolumeFilter"
timeframe = "12h"
leverage = 1.0