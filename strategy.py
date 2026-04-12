#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_keltner_squeeze_breakout_v1"
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
    
    # Get 1d data for Keltner channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(20) and EMA(20) for Keltner channels
    tr_1d = np.maximum(high_1d[1:] - low_1d[1:], 
                       np.maximum(np.abs(high_1d[1:] - close_1d[:-1]), 
                                  np.abs(low_1d[1:] - close_1d[:-1])))
    tr_1d = np.concatenate([[np.nan], tr_1d])  # First value NaN
    
    tr_series = pd.Series(tr_1d)
    atr_20_1d = tr_series.rolling(window=20, min_periods=20).mean().values
    
    close_1d_series = pd.Series(close_1d)
    ema_20_1d = close_1d_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner channels: upper = EMA + 2*ATR, lower = EMA - 2*ATR
    keltner_upper_1d = ema_20_1d + 2.0 * atr_20_1d
    keltner_lower_1d = ema_20_1d - 2.0 * atr_20_1d
    
    # Align Keltner channels to 4h timeframe
    keltner_upper_aligned = align_htf_to_ltf(prices, df_1d, keltner_upper_1d)
    keltner_lower_aligned = align_htf_to_ltf(prices, df_1d, keltner_lower_1d)
    ema_20_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Volume filter: current volume > 20-period average (on 4h data)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma
    
    # Bollinger Bands on 4h for squeeze detection (20, 2)
    bb_length = 20
    bb_mult = 2.0
    close_series = pd.Series(close)
    basis = close_series.rolling(window=bb_length, min_periods=bb_length).mean().values
    dev = bb_mult * close_series.rolling(window=bb_length, min_periods=bb_length).std().values
    upper_bb = basis + dev
    lower_bb = basis - dev
    bb_width = (upper_bb - lower_bb) / basis
    
    # Bollinger Band width percentile (lookback 50 periods) to detect squeeze
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 0 else np.nan, raw=False
    ).values
    squeeze_condition = bb_width_percentile < 20  # Squeeze when BB width is in lowest 20%
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # warmup for indicators
        # Skip if not ready
        if (np.isnan(keltner_upper_aligned[i]) or np.isnan(keltner_lower_aligned[i]) or
            np.isnan(ema_20_aligned[i]) or np.isnan(volume_ok[i]) or np.isnan(squeeze_condition[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Squeeze breakout conditions
        breakout_up = close[i] > keltner_upper_aligned[i] and squeeze_condition[i]
        breakout_down = close[i] < keltner_lower_aligned[i] and squeeze_condition[i]
        
        # Volume confirmation
        vol_ok = volume_ok[i]
        
        # Trend filter: price relative to 20-period EMA on 1d
        uptrend = close[i] > ema_20_aligned[i]
        downtrend = close[i] < ema_20_aligned[i]
        
        # Entry signals
        long_signal = breakout_up and vol_ok and uptrend
        short_signal = breakout_down and vol_ok and downtrend
        
        # Exit when price returns to the 20-period EMA (mean reversion)
        exit_long = close[i] < ema_20_aligned[i]
        exit_short = close[i] > ema_20_aligned[i]
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals