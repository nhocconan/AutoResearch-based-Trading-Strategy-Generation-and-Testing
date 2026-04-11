#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Bollinger Band squeeze breakout with 1d volume confirmation and ADX trend filter.
# In low volatility regimes (BB width < 20th percentile), price is poised for breakout.
# Enter long when price breaks above upper BB with volume > 1.5x average and ADX > 25.
# Enter short when price breaks below lower BB with volume > 1.5x average and ADX > 25.
# Exit when price re-enters the Bollinger Bands or volatility expands (BB width > 50th percentile).
# Designed for 15-25 trades/year on 12h timeframe with focus on volatility breakouts.

name = "12h_1d_bb_squeeze_breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d average volume for confirmation
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate Bollinger Bands (20, 2) on 12h
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    
    # Calculate BB width percentile (20-period lookback)
    bb_width = (upper_bb - lower_bb) / sma_20
    # Use 50-period lookback for percentile to avoid look-ahead
    bb_width_rank = pd.Series(bb_width).rolling(window=50, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5, raw=False
    ).values
    
    # Calculate ADX (14) on 12h for trend filter
    # TR = max(high-low, abs(high-close_prev), abs(low-close_prev))
    high_low = high - low
    high_close_prev = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    low_close_prev = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # +DM and -DM
    up_move = high - np.concatenate([[high[0]], high[:-1]])
    down_move = np.concatenate([[low[0]], low[:-1]]) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed +DM, -DM, TR
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    tr_smooth = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # +DI and -DI
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after sufficient data for indicators
        # Skip if any required data is invalid
        if (np.isnan(upper_bb[i]) or np.isnan(lower_bb[i]) or 
            np.isnan(bb_width_rank[i]) or np.isnan(adx[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volatility regime: BB width rank < 0.2 (low volatility squeeze)
        vol_squeeze = bb_width_rank[i] < 0.2
        
        # Volume confirmation: current volume > 1.5 * 20-period average volume (1d)
        vol_confirm = volume[i] > 1.5 * vol_ma_20_1d_aligned[i]
        
        # Trend filter: ADX > 25
        trend_filter = adx[i] > 25
        
        # Breakout conditions
        breakout_up = close[i] > upper_bb[i]
        breakout_down = close[i] < lower_bb[i]
        
        # Entry conditions
        long_entry = vol_squeeze and vol_confirm and trend_filter and breakout_up
        short_entry = vol_squeeze and vol_confirm and trend_filter and breakout_down
        
        # Exit conditions: volatility expansion OR mean reversion
        vol_expansion = bb_width_rank[i] > 0.5  # BB width > 50th percentile
        mean_reversion = (position == 1 and close[i] < sma_20[i]) or \
                         (position == -1 and close[i] > sma_20[i])
        
        # Priority: exit > entry > hold
        if position == 1 and (vol_expansion or mean_reversion):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (vol_expansion or mean_reversion):
            position = 0
            signals[i] = 0.0
        elif long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals