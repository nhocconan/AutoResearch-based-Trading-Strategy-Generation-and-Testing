#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1-week Keltner Channel breakout with volume confirmation and ADX trend filter
# Long when price closes above weekly Keltner upper band with volume > 1.5x average and ADX > 25
# Short when price closes below weekly Keltner lower band with volume > 1.5x average and ADX > 25
# Weekly Keltner Channel adapts to volatility, providing dynamic support/resistance.
# Volume confirms breakout strength, ADX filters for trending conditions to avoid whipsaws.
# Designed to work in both bull and bear markets by capturing strong momentum moves.
# Target: 15-25 trades per year (60-100 over 4 years) with 0.25 position sizing.

name = "1d_1wKeltner_20_2.0_ADXVol_Breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1-week Keltner Channel (20, 2.0) ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Typical Price for Keltner Channel
    tp = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    
    # EMA of Typical Price (20-period)
    ema_tp = tp.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Average True Range (20-period)
    tr_w = pd.DataFrame({
        'hl': df_1w['high'] - df_1w['low'],
        'hc': abs(df_1w['high'] - df_1w['close'].shift(1)),
        'lc': abs(df_1w['low'] - df_1w['close'].shift(1))
    }).max(axis=1)
    atr_w = tr_w.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner Channel bands
    keltner_upper = ema_tp + (2.0 * atr_w)
    keltner_lower = ema_tp - (2.0 * atr_w)
    
    # Align weekly Keltner levels to daily timeframe
    keltner_upper_aligned = align_htf_to_ltf(prices, df_1w, keltner_upper)
    keltner_lower_aligned = align_htf_to_ltf(prices, df_1w, keltner_lower)
    
    # Calculate daily ADX (14-period) for trend strength
    # True Range
    tr_d = pd.DataFrame({
        'hl': high - low,
        'hc': np.abs(high - np.append([np.nan], close[:-1])),
        'lc': np.abs(low - np.append([np.nan], close[:-1]))
    }).max(axis=1)
    
    # Directional Movement
    up_move = np.append([np.nan], high[1:] - high[:-1])
    down_move = np.append([np.nan], low[:-1] - low[1:])
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_period = 14
    atr_d = pd.Series(tr_d).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr_d
    minus_di = 100 * minus_dm_smooth / atr_d
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    
    # Align ADX to daily timeframe (already daily, but using align for consistency)
    adx_aligned = align_htf_to_ltf(prices, prices, adx)
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Signals array
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any critical value is NaN
        if (np.isnan(keltner_upper_aligned[i]) or np.isnan(keltner_lower_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price closes above weekly Keltner upper with volume and trend
            if close[i] > keltner_upper_aligned[i] and volume_filter[i] and adx_aligned[i] > 25:
                signals[i] = 0.25
                position = 1
            # Short breakout: price closes below weekly Keltner lower with volume and trend
            elif close[i] < keltner_lower_aligned[i] and volume_filter[i] and adx_aligned[i] > 25:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below weekly Keltner lower (trend reversal)
            if close[i] < keltner_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above weekly Keltner upper (trend reversal)
            if close[i] > keltner_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals