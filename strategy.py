#!/usr/bin/env python3

"""
Hypothesis: 4-hour Bollinger Band breakout with daily trend filter and volume confirmation.
Only trade long when price breaks above upper Bollinger Band with daily trend up and volume spike;
short when price breaks below lower Bollinger Band with daily trend down and volume spike.
Exit when price returns to middle band or volatility expands. Designed for moderate trade frequency
(15-35 trades/year) by requiring multiple confirmations: volatility breakout, trend alignment, and volume.
Works in both bull and bear markets by following the daily trend.
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
    
    # Bollinger Bands (20, 2) on 4h
    close_s = pd.Series(close)
    bb_middle = close_s.rolling(window=20, min_periods=20).mean().values
    bb_std = close_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb.std
    bb_width = bb_upper - bb_lower
    
    # Bollinger Band width percentile (50-period lookback) for volatility filter
    bb_width_pct = pd.Series(bb_width).rolling(window=50, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False
    ).values
    
    # Load daily data for trend filter - ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 10:
        return np.zeros(n)
    
    # Daily EMA34 for trend direction
    daily_close = df_daily['close'].values
    ema34_daily = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_daily_aligned = align_htf_to_ltf(prices, df_daily, ema34_daily)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(bb_width_pct[i]) or np.isnan(bb_upper[i]) or 
            np.isnan(bb_lower[i]) or np.isnan(ema34_daily_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility expansion filter: avoid breakouts in high volatility
        vol_expansion = bb_width_pct[i] > 0.7
        
        # Volume confirmation
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: volatility contraction + price breaks above upper band + daily uptrend + volume spike
            if (not vol_expansion) and close[i] > bb_upper[i] and ema34_daily_aligned[i] > ema34_daily_aligned[i-1] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: volatility contraction + price breaks below lower band + daily downtrend + volume spike
            elif (not vol_expansion) and close[i] < bb_lower[i] and ema34_daily_aligned[i] < ema34_daily_aligned[i-1] and vol_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: volatility expansion or price returns to middle band
            exit_signal = False
            
            if position == 1:
                # Exit long: volatility expansion or price closes below middle band
                if vol_expansion or close[i] < bb_middle[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: volatility expansion or price closes above middle band
                if vol_expansion or close[i] > bb_middle[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Bollinger_Breakout_DailyTrend_Volume"
timeframe = "4h"
leverage = 1.0