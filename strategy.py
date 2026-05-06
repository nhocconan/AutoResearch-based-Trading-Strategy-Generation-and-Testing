#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Donchian breakout with volume confirmation and ADX trend filter
# Long when price breaks above weekly Donchian upper with volume > 1.8x average and ADX > 25 (strong trend)
# Short when price breaks below weekly Donchian lower with volume > 1.8x average and ADX > 25 (strong trend)
# Weekly Donchian provides robust trend-following structure. Volume confirms breakout momentum.
# ADX filter ensures trades occur only in trending markets, avoiding whipsaws in ranging conditions.
# Works in bull/bear markets: captures sustained moves while filtering false breakouts.
# Target: 15-25 trades per year (60-100 over 4 years) with 0.30 position sizing.

name = "1d_weeklyDonchian_TrendVol_Filter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate weekly Donchian channels (20-period) ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly Donchian upper and lower (20-period high/low)
    donchian_high = pd.Series(df_1w['high']).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_1w['low']).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Weekly ADX for trend filter (14-period)
    if len(df_1w) >= 14:
        # Calculate True Range
        high_low = df_1w['high'] - df_1w['low']
        high_close = np.abs(df_1w['high'] - df_1w['close'].shift(1))
        low_close = np.abs(df_1w['low'] - df_1w['close'].shift(1))
        tr = np.maximum(np.maximum(high_low, high_close), low_close)
        
        # Calculate Directional Movement
        up_move = df_1w['high'] - df_1w['high'].shift(1)
        down_move = df_1w['low'].shift(1) - df_1w['low']
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smooth TR and DM (Wilder's smoothing = EMA with alpha=1/period)
        atr = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
        plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values / atr
        minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values / atr
        
        # Calculate DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
        adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    else:
        adx_aligned = np.full(n, np.nan)
    
    # Volume confirmation: >1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.8 * vol_ma_20)
    
    # Pre-compute session filter (00-24 UTC for daily - full day)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = np.ones(n, dtype=bool)  # Trade all hours for daily timeframe
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above weekly Donchian high with volume and trend confirmation
            if close[i] > donchian_high_aligned[i] and volume_filter[i] and adx_aligned[i] > 25:
                signals[i] = 0.30
                position = 1
            # Short breakout: price breaks below weekly Donchian low with volume and trend confirmation
            elif close[i] < donchian_low_aligned[i] and volume_filter[i] and adx_aligned[i] > 25:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price breaks below weekly Donchian low (trend reversal) or ADX weakens
            if close[i] < donchian_low_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price breaks above weekly Donchian high (trend reversal) or ADX weakens
            if close[i] > donchian_high_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals