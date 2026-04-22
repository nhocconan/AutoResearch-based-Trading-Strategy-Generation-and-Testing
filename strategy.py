#!/usr/bin/env python3
"""
Hypothesis: 12-hour Bollinger Band squeeze with daily trend filter and volume confirmation.
Long when price breaks above upper BB during low volatility (BBW < 20th percentile) and price > daily EMA50.
Short when price breaks below lower BB during low volatility (BBW < 20th percentile) and price < daily EMA50.
Exit when price returns to middle BB (mean reversion) or volatility expands (BBW > 80th percentile).
BB squeeze captures low volatility breakouts; daily EMA filter ensures trend alignment; volume avoids false breakouts.
Works in both bull and bear markets by trading volatility contractions/expansions with trend alignment.
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
    
    # Bollinger Bands (20, 2) - calculated on 12h data
    bb_period = 20
    bb_std = 2.0
    
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_bb = sma + bb_std * std
    lower_bb = sma - bb_std * std
    middle_bb = sma
    bb_width = (upper_bb - lower_bb) / middle_bb  # Normalized bandwidth
    
    # Percentile of BB width for squeeze detection (lookback 50 periods)
    bb_width_series = pd.Series(bb_width)
    bb_width_pct = bb_width_series.rolling(window=50, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Daily EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Daily volume average for confirmation
    vol_1d = df_1d['volume'].values
    avg_vol_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    avg_vol_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(upper_bb[i]) or np.isnan(lower_bb[i]) or 
            np.isnan(bb_width_pct[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(avg_vol_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: BB squeeze (low volatility) + breakout above upper BB + price > daily EMA50 + volume confirmation
            if (bb_width_pct[i] < 20 and  # Bollinger Band squeeze (low volatility)
                close[i] > upper_bb[i] and  # Break above upper BB
                close[i] > ema_50_1d_aligned[i] and  # Above daily EMA50 (uptrend)
                volume[i] > avg_vol_1d_aligned[i]):  # Volume above average
                signals[i] = 0.25
                position = 1
            # Short: BB squeeze + breakdown below lower BB + price < daily EMA50 + volume confirmation
            elif (bb_width_pct[i] < 20 and  # Bollinger Band squeeze (low volatility)
                  close[i] < lower_bb[i] and  # Break below lower BB
                  close[i] < ema_50_1d_aligned[i] and  # Below daily EMA50 (downtrend)
                  volume[i] > avg_vol_1d_aligned[i]):  # Volume above average
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price returns to middle BB OR volatility expands (BBW > 80th percentile)
                if (close[i] < middle_bb[i] or bb_width_pct[i] > 80):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price returns to middle BB OR volatility expands
                if (close[i] > middle_bb[i] or bb_width_pct[i] > 80):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_BB_Squeeze_DailyEMA50_Volume"
timeframe = "12h"
leverage = 1.0