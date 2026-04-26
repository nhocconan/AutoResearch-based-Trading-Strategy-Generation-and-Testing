#!/usr/bin/env python3
"""
4h_Donchian20_TrendFilter_VolumeSpike_ATRStop
Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation (>1.8x 20-period average). Uses ATR(14) trailing stop (2.0x) and Bollinger Band width regime filter to avoid low-volatility whipsaws. Discrete sizing 0.30 targets ~25 trades/year (100 total) to minimize fee drag. Works in bull/bear markets: trend filter adapts to 1d momentum, volume ensures conviction, regime filter avoids chop.
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
    open_time = prices['open_time'].values
    
    # Session filter: UTC 8-20 for institutional activity
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # ATR(14) on 4h for breakout confirmation and trailing stop
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Bollinger Band width (20,2) on 4h for regime filter - avoids low volatility whipsaws
    bb_ma = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_ma + 2 * bb_std
    bb_lower = bb_ma - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_ma  # Normalized width
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    low_volatility = bb_width < 0.5 * bb_width_ma  # Avoid trading in extremely low vol
    
    # Donchian(20) channels on 4h
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup: max of Donchian (20), 1d EMA (50), ATR (14), BB width MA (20)
    start_idx = max(20, 50, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session or low volatility regime
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or 
            np.isnan(vol_ma[i]) or
            np.isnan(atr[i]) or
            np.isnan(bb_width[i]) or
            np.isnan(bb_width_ma[i]) or
            not in_session[i] or
            low_volatility[i]):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.30
            else:
                signals[i] = -0.30
            continue
        
        ema_50_1d_val = ema_50_1d_aligned[i]
        upper_channel = highest_20[i]
        lower_channel = lowest_20[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        atr_val = atr[i]
        
        # Volume confirmation: current volume > 1.8x 20-period average (moderate for quality)
        volume_confirmed = vol_val > 1.8 * vol_ma_val
        # Breakout threshold: price must close beyond Donchian level by 0.5*ATR (moderate)
        breakout_threshold = 0.5 * atr_val
        
        if position == 0:
            # Long: close above upper channel + threshold, uptrend (close > EMA50_1d), volume confirmation
            long_signal = (close_val > upper_channel + breakout_threshold) and (close_val > ema_50_1d_val) and volume_confirmed
            # Short: close below lower channel - threshold, downtrend (close < EMA50_1d), volume confirmation
            short_signal = (close_val < lower_channel - breakout_threshold) and (close_val < ema_50_1d_val) and volume_confirmed
            
            if long_signal:
                signals[i] = 0.30
                position = 1
                entry_price = close_val
                highest_since_entry = close_val
            elif short_signal:
                signals[i] = -0.30
                position = -1
                entry_price = close_val
                lowest_since_entry = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.30
            highest_since_entry = max(highest_since_entry, high_val)
            # ATR trailing stop: exit if price drops 2.0*ATR from high (tighter stop for trend)
            if close_val < highest_since_entry - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
            # Exit: price closes below lower channel
            elif close_val < lower_channel:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
            # Exit: trend reversal (close below EMA50_1d)
            elif close_val < ema_50_1d_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
        elif position == -1:
            # Hold short
            signals[i] = -0.30
            lowest_since_entry = min(lowest_since_entry, low_val)
            # ATR trailing stop: exit if price rises 2.0*ATR from low
            if close_val > lowest_since_entry + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                lowest_since_entry = 0.0
            # Exit: price closes above upper channel
            elif close_val > upper_channel:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                lowest_since_entry = 0.0
            # Exit: trend reversal (close above EMA50_1d)
            elif close_val > ema_50_1d_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                lowest_since_entry = 0.0
    
    return signals

name = "4h_Donchian20_TrendFilter_VolumeSpike_ATRStop"
timeframe = "4h"
leverage = 1.0