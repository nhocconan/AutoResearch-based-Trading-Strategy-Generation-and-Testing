#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme + 1d ADX trend filter + volume spike confirmation.
# Long when Williams %R < -80 (oversold) AND 1d ADX > 25 (trending) AND volume > 2.0x 20-period median.
# Short when Williams %R > -20 (overbought) AND 1d ADX > 25 (trending) AND volume > 2.0x 20-period median.
# Williams %R identifies exhaustion points in trends; ADX ensures we trade with the trend, not against it.
# Volume spike confirms conviction at extremes. Works in bull markets (buy oversold in uptrend) and bear markets (sell overbought in downtrend).
# Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years).

name = "6h_WilliamsR_Extreme_1dADX_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period volume median for volume confirmation
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # Calculate Williams %R(14) - using prior bar's data to avoid look-ahead
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # Calculate 1d ADX(14) trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate ADX components
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    
    # True Range
    tr1 = df_1d_high[1:] - df_1d_low[1:]
    tr2 = np.abs(df_1d_high[1:] - df_1d_close[:-1])
    tr3 = np.abs(df_1d_low[1:] - df_1d_close[:-1])
    tr_first = np.max([df_1d_high[0] - df_1d_low[0], np.abs(df_1d_high[0] - df_1d_close[0]), np.abs(df_1d_low[0] - df_1d_close[0])])
    tr_1d = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((df_1d_high[1:] - df_1d_high[:-1]) > (df_1d_low[:-1] - df_1d_low[1:]), 
                       np.maximum(df_1d_high[1:] - df_1d_high[:-1], 0), 0)
    dm_minus = np.where((df_1d_low[:-1] - df_1d_low[1:]) > (df_1d_high[1:] - df_1d_high[:-1]), 
                        np.maximum(df_1d_low[:-1] - df_1d_low[1:], 0), 0)
    # Handle first bar
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+, DM-
    tr_14 = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    # Handle division by zero
    dx = np.where((di_plus + di_minus) == 0, 0, dx)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # Align HTF indicators to LTF
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for ATR, Williams %R, ADX, and volume
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or 
            np.isnan(williams_r_aligned[i]) or 
            np.isnan(adx_aligned[i]) or 
            np.isnan(vol_median_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Williams %R conditions
        oversold = williams_r_aligned[i] < -80
        overbought = williams_r_aligned[i] > -20
        
        # ADX trend filter: trending market
        trending = adx_aligned[i] > 25
        
        # Volume confirmation: current volume > 2.0x 20-period volume median
        if vol_median_20[i] <= 0 or np.isnan(vol_median_20[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_20[i] * 2.0)
        
        if position == 0:  # Flat - look for new entries
            # Long: oversold AND trending AND volume spike
            if oversold and trending and volume_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: overbought AND trending AND volume spike
            elif overbought and trending and volume_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Stoploss: price moves against position by 2.0*ATR
            if curr_close < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Williams %R rises above -50 (exit oversold) OR ADX weakens
            elif williams_r_aligned[i] > -50 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: price moves against position by 2.0*ATR
            if curr_close > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Williams %R falls below -50 (exit overbought) OR ADX weakens
            elif williams_r_aligned[i] < -50 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals