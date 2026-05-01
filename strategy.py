#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme reversal with 12h ADX trend filter and volume confirmation.
# Long when Williams %R < -80 (oversold) AND 12h ADX > 25 (trending) AND volume > 1.5x 20-period median.
# Short when Williams %R > -20 (overbought) AND 12h ADX > 25 AND volume > 1.5x 20-period median.
# Uses discrete sizing 0.25. ATR(14) stoploss: signal→0 when price moves against position by 2.0*ATR.
# Target: 12-30 trades/year on 6h timeframe (~50-120 total over 4 years).
# Williams %R captures exhaustion moves; ADX filters choppy markets; volume confirms conviction.
# Works in both bull/bear markets by trading mean reversions within established trends.

name = "6h_WilliamsR_Extreme_12hADX_Volume_v1"
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
    
    # Calculate Williams %R(14) - looks back 14 periods
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # Calculate 12h ADX(14) trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    # True Range for ADX
    tr_12h = np.maximum(
        df_12h['high'].values - df_12h['low'].values,
        np.maximum(
            np.abs(df_12h['high'].values - np.concatenate([[df_12h['close'].values[0]], df_12h['close'].values[:-1]])),
            np.abs(df_12h['low'].values - np.concatenate([[df_12h['close'].values[0]], df_12h['close'].values[:-1]]))
        )
    )
    # Directional Movement
    dm_plus = np.where(
        (df_12h['high'].values - np.concatenate([[df_12h['high'].values[0]], df_12h['high'].values[:-1]])) > 
        (np.concatenate([[df_12h['low'].values[0]], df_12h['low'].values[:-1]]) - df_12h['low'].values),
        np.maximum(df_12h['high'].values - np.concatenate([[df_12h['high'].values[0]], df_12h['high'].values[:-1]]), 0),
        0
    )
    dm_minus = np.where(
        (np.concatenate([[df_12h['low'].values[0]], df_12h['low'].values[:-1]]) - df_12h['low'].values) > 
        (df_12h['high'].values - np.concatenate([[df_12h['high'].values[0]], df_12h['high'].values[:-1]])),
        np.maximum(np.concatenate([[df_12h['low'].values[0]], df_12h['low'].values[:-1]]) - df_12h['low'].values, 0),
        0
    )
    # Smoothed values
    tr_14 = pd.Series(tr_12h).rolling(window=14, min_periods=14).sum().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / np.where(tr_14 == 0, 1, tr_14)
    di_minus = 100 * dm_minus_14 / np.where(tr_14 == 0, 1, tr_14)
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / np.where((di_plus + di_minus) == 0, 1, (di_plus + di_minus))
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for Williams %R, ADX, ATR, and volume
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(williams_r[i]) or 
            np.isnan(adx_aligned[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(vol_median_20[i]) or 
            vol_median_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Trend filter: 12h ADX > 25 indicates trending market
        trending = adx_aligned[i] > 25
        
        # Volume confirmation: current volume > 1.5x 20-period median
        volume_confirm = curr_volume > (vol_median_20[i] * 1.5)
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R < -80 (oversold) AND trending AND volume spike
            if williams_r[i] < -80 and trending and volume_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Williams %R > -20 (overbought) AND trending AND volume spike
            elif williams_r[i] > -20 and trending and volume_confirm:
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
            # Exit: Williams %R > -50 (exiting oversold) OR ADX < 20 (losing trend)
            elif williams_r[i] > -50 or adx_aligned[i] < 20:
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
            # Exit: Williams %R < -50 (exiting overbought) OR ADX < 20 (losing trend)
            elif williams_r[i] < -50 or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals