#!/usr/bin/env python3
name = "4h_Camarilla_R3S3_Breakout_1dTrend_Volume_Enhanced"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    # Camarilla R3 and S3 levels
    r3 = close_prev + 1.1 * (high_prev - low_prev) / 6
    s3 = close_prev - 1.1 * (high_prev - low_prev) / 6
    
    # Align daily levels to 4h timeframe (with 1-day delay for completed bar)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Daily trend filter: EMA34
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: current volume > 1.5x 20-period average (4h)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (1.5 * vol_ma_20)
    
    # Choppiness regime filter (1d CHOP > 61.8 = range, < 38.2 = trend)
    # Calculate True Range for 1d
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14, min_periods=14).mean().values
    # Calculate ADX-like component for chop: |close - open| / ATR
    body_size = abs(df_1d['close'] - df_1d['open']).values
    chop_raw = 100 * body_size / atr_14
    chop = pd.Series(chop_raw).rolling(window=14, min_periods=14).mean().values
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 3  # ~6 hours for 4h to reduce trades
    
    start_idx = max(100, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or
            np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine daily trend direction
        trend_up = close > ema_34_1d_aligned[i]
        trend_down = close < ema_34_1d_aligned[i]
        
        # Chop regime: only trade when market is trending (CHOP < 38.2)
        is_trending = chop_aligned[i] < 38.2
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Break above R3 in uptrend with volume and trending regime
            if (close[i] > r3_aligned[i] and 
                trend_up[i] and 
                vol_filter[i] and
                is_trending):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Break below S3 in downtrend with volume and trending regime
            elif (close[i] < s3_aligned[i] and 
                  trend_down[i] and 
                  vol_filter[i] and
                  is_trending):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Price re-enters Camarilla body (between R3 and S3) or trend change or chop regime
            if ((close[i] < r3_aligned[i] and close[i] > s3_aligned[i]) or 
                not trend_up[i] or
                chop_aligned[i] >= 61.8):  # choppy market, exit
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price re-enters Camarilla body or trend change or chop regime
            if ((close[i] < r3_aligned[i] and close[i] > s3_aligned[i]) or 
                not trend_down[i] or
                chop_aligned[i] >= 61.8):  # choppy market, exit
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Enhanced Camarilla R3/S3 breakout strategy with chop regime filter.
# Only takes trades in trending markets (CHOP < 38.2) and exits when choppy (CHOP >= 61.8).
# This reduces whipsaws in sideways markets while capturing strong trends.
# Volume confirmation ensures institutional participation. Target: 20-35 trades/year.