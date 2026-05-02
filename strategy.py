#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Squeeze Breakout with 1d ADX Trend Filter and Volume Spike
# Bollinger Band Squeeze (low volatility) precedes explosive moves in both bull and bear markets
# Breakout above upper BB or below lower BB with volume confirmation captures the move
# 1d ADX > 20 ensures alignment with trending market to avoid false breakouts in ranges
# Uses discrete position sizing (0.25) to minimize fee churn and control drawdown
# Target: 12-37 trades/year (50-150 total over 4 years) on 6h timeframe

name = "6h_BollingerSqueeze_Breakout_1dADX_Trend_Volume"
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
    
    # 1d data for trend filter (ADX)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    # 1d ADX calculation (using standard Wilder's smoothing)
    # True Range
    tr1 = pd.Series(df_1d['high']).diff().abs()
    tr2 = pd.Series(df_1d['low']).diff().abs()
    tr3 = (pd.Series(df_1d['close']).shift() - pd.Series(df_1d['high'])).abs()
    tr4 = (pd.Series(df_1d['close']).shift() - pd.Series(df_1d['low'])).abs()
    tr = pd.concat([tr1, tr2, tr3, tr4], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    
    # Directional Movement
    up_move = pd.Series(df_1d['high']).diff()
    down_move = -pd.Series(df_1d['low']).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean()
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean()
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean()
    adx_values = adx.values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    
    # Bollinger Bands (20, 2) on 6h close
    close_s = pd.Series(close)
    bb_middle = close_s.rolling(window=20, min_periods=20).mean().values
    bb_std = close_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_middle  # Normalized width
    
    # Bollinger Band Squeeze: width below 20-period rolling mean of width
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    bb_squeeze = bb_width < bb_width_ma
    
    # Volume confirmation: spike above 2.0 * 20-period EMA
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmation = volume > (2.0 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 40  # Need 20 for BB + 20 for BB width MA + buffer
    
    for i in range(start_idx, n):
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(bb_squeeze[i]) or 
            np.isnan(volume_confirmation[i]) or np.isnan(bb_upper[i]) or 
            np.isnan(bb_lower[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 1d ADX (trending market filter)
        trending_market = adx_1d_aligned[i] > 20
        
        if position == 0:  # Flat - look for new entries
            # Long: BB squeeze breakout above upper BB with volume confirmation and trending market
            if bb_squeeze[i-1] and close[i] > bb_upper[i] and trending_market and volume_confirmation[i]:
                signals[i] = 0.25
                position = 1
            # Short: BB squeeze breakout below lower BB with volume confirmation and trending market
            elif bb_squeeze[i-1] and close[i] < bb_lower[i] and trending_market and volume_confirmation[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price closes below middle BB (mean reversion) OR market loses trend
            if close[i] < bb_middle[i] or not trending_market:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price closes above middle BB (mean reversion) OR market loses trend
            if close[i] > bb_middle[i] or not trending_market:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals