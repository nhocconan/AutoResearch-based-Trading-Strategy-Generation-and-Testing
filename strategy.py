#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d EMA50 trend filter and volume confirmation
# Elder Ray measures bull/bear power relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# In trending markets (1d ADX > 25): 
#   - Bull market: Buy on Bull Power > 0 with volume confirmation
#   - Bear market: Sell on Bear Power < 0 with volume confirmation
# Uses discrete sizing (0.25) to minimize fees. Target: 12-37 trades/year (50-150 total over 4 years)
# Works in bull markets (buy strength) and bear markets (sell weakness)

name = "6h_ElderRay_1dADX_Trend_Volume"
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
    
    # 1d data for trend filter (ADX) and EMA13 for Elder Ray
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for ADX/EMA calculation
        return np.zeros(n)
    
    # 1d ADX calculation (Wilder's smoothing)
    tr1 = pd.Series(df_1d['high']).diff().abs()
    tr2 = pd.Series(df_1d['low']).diff().abs()
    tr3 = (pd.Series(df_1d['close']).shift() - pd.Series(df_1d['high'])).abs()
    tr4 = (pd.Series(df_1d['close']).shift() - pd.Series(df_1d['low'])).abs()
    tr = pd.concat([tr1, tr2, tr3, tr4], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    
    up_move = pd.Series(df_1d['high']).diff()
    down_move = -pd.Series(df_1d['low']).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean()
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean()
    
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean()
    adx_values = adx.values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    
    # 1d EMA13 for Elder Ray calculation
    ema13_1d = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    
    # 6h EMA13 for Elder Ray (same period as HTF for consistency)
    close_s = pd.Series(close)
    ema13_6h = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13_6h
    bear_power = low - ema13_6h
    
    # Volume confirmation (2x EMA20 threshold for institutional participation)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmation = volume > (2.0 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(ema13_1d_aligned[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 1d ADX (trending market filter)
        trending_market = adx_1d_aligned[i] > 25
        
        if position == 0:  # Flat - look for new entries
            # Long: Bull Power > 0 (strength) with volume confirmation and trending market
            if bull_power[i] > 0 and trending_market and volume_confirmation[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 (weakness) with volume confirmation and trending market
            elif bear_power[i] < 0 and trending_market and volume_confirmation[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Bull Power turns negative (loss of strength) OR market loses trend
            if bull_power[i] <= 0 or not trending_market:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Bear Power turns positive (loss of weakness) OR market loses trend
            if bear_power[i] >= 0 or not trending_market:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals