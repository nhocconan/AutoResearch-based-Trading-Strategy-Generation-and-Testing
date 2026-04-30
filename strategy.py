#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA200 trend filter, volume confirmation, and chop regime filter.
# Donchian breakouts capture strong momentum moves. 1d EMA200 ensures we only trade with the higher timeframe trend.
# Volume confirmation (>1.5x 20-bar avg) reduces false breakouts. Chop regime filter (CHOP > 61.8) avoids ranging markets.
# ATR-based trailing stop (2.0x ATR) manages risk. Discrete position sizing at ±0.30 balances capture and fee drag.
# Target: 100-180 total trades over 4 years (25-45/year) to stay within healthy limits.
# Works in both bull and bear markets by aligning with 1d trend and using volatility-based stops.

name = "4h_Donchian20_1dEMA200_VolumeChop_ATRStop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid look-ahead
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1d data ONCE before loop for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 210:  # Need enough for EMA200
        return np.zeros(n)
    
    # Calculate 1d EMA200 for trend filter
    close_1d_vals = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d_vals).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Load 1d data ONCE before loop for chop regime filter (using 14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    
    # True Range for chop calculation
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d_vals[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d_vals[:-1])
    tr_1d = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Chopiness Index: CHOP = 100 * log10(sum(ATR14) / (max(high) - min(low))) / log10(14)
    # We'll use a simplified version: CHOP = 100 * (ATR14 / (HHV14 - LLV14))
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denom = hh_14 - ll_14
    # Avoid division by zero
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)
    chop_1d = 100 * (atr_1d / chop_denom)
    # Chop > 61.8 indicates ranging market (we want to avoid this for breakout strategy)
    chop_regime = chop_1d <= 61.8  # Trending when CHOP <= 61.8
    
    # Align 1d indicators to 4h timeframe
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    chop_regime_aligned = align_htf_to_ltf(prices, df_1d, chop_regime.astype(float))
    
    # Donchian channels (20-period) on 4h data
    donchian_period = 20
    upper_channel = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower_channel = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # ATR(14) for volatility and stoploss on 4h data
    atr_period = 14
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = 200  # warmup for EMA200 and indicators
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session or choppy regime
        if (np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(chop_regime_aligned[i]) or
            np.isnan(upper_channel[i]) or
            np.isnan(lower_channel[i]) or
            np.isnan(atr[i]) or
            np.isnan(volume_confirm[i]) or
            not in_session[i] or
            chop_regime_aligned[i] == 0.0):  # Only trade in trending regime (CHOP <= 61.8)
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_upper = upper_channel[i]
        curr_lower = lower_channel[i]
        curr_ema_200_1d = ema_200_1d_aligned[i]
        curr_atr = atr[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian upper, above 1d EMA200, volume spike, trending regime
            if (curr_close > curr_upper and 
                curr_close > curr_ema_200_1d and 
                curr_volume_confirm):
                signals[i] = 0.30
                position = 1
                entry_price = curr_close
                highest_since_entry = curr_close
            # Short: price breaks below Donchian lower, below 1d EMA200, volume spike, trending regime
            elif (curr_close < curr_lower and 
                  curr_close < curr_ema_200_1d and 
                  curr_volume_confirm):
                signals[i] = -0.30
                position = -1
                entry_price = curr_close
                lowest_since_entry = curr_close
        
        elif position == 1:  # Long position
            # Update highest price since entry
            highest_since_entry = max(highest_since_entry, curr_high)
            # ATR trailing stop: exit if price drops 2.0*ATR from highest point
            if curr_close < highest_since_entry - (2.0 * curr_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Update lowest price since entry
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # ATR trailing stop: exit if price rises 2.0*ATR from lowest point
            if curr_close > lowest_since_entry + (2.0 * curr_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals