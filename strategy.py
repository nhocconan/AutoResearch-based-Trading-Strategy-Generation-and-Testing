#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly EMA trend filter, daily Donchian breakout with volume confirmation
# Weekly EMA(34) determines long-term trend direction (bull/bear/range)
# Daily Donchian(20) breakout in direction of weekly trend captures momentum moves
# Volume confirmation (current 6h volume > 1.5x 20-period average) filters false breakouts
# Fixed position size 0.25 to balance return and drawdown
# Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)

name = "6h_1w_1d_donchian_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Load 1w and 1d data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 10 or len(df_1d) < 10:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly EMA(34) for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate daily Donchian channels (20-period)
    high_max_20_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_min_20_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate daily ATR(14) for volatility filtering (optional)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = tr1_1d[0]  # First period has no previous close
    atr_14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Align all HTF data to 6h timeframe
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    high_max_20_1d_aligned = align_htf_to_ltf(prices, df_1d, high_max_20_1d)
    low_min_20_1d_aligned = align_htf_to_ltf(prices, df_1d, low_min_20_1d)
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Pre-compute volume confirmation (20-period average for 6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid or outside session
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(high_max_20_1d_aligned[i]) or
            np.isnan(low_min_20_1d_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or
            np.isnan(vol_ma_20[i]) or not in_session[i] or
            atr_14_1d_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x average 6h volume
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if not volume_confirmed:
            signals[i] = 0.0
            continue
        
        # Fixed position size
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit on retracement to weekly EMA or breakdown of daily Donchian low
            if close[i] < ema_34_1w_aligned[i] or close[i] < low_min_20_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit on retracement to weekly EMA or breakout above daily Donchian high
            if close[i] > ema_34_1w_aligned[i] or close[i] > high_max_20_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Only trade in direction of weekly trend
            # Weekly trend: price above EMA = bullish, below EMA = bearish
            if close[i] > ema_34_1w_aligned[i]:  # Weekly bullish trend
                # Long breakout above daily Donchian high
                if close[i] > high_max_20_1d_aligned[i]:
                    position = 1
                    signals[i] = position_size
            else:  # Weekly bearish trend
                # Short breakdown below daily Donchian low
                if close[i] < low_min_20_1d_aligned[i]:
                    position = -1
                    signals[i] = -position_size
    
    return signals