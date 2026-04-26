#!/usr/bin/env python3
"""
12h_Camarilla_R3S3_Breakout_1dTrend_ADXFilter
Hypothesis: Camarilla R3/S3 levels act as strong intraday support/resistance on 12h timeframe. 
Breakout above R3 with volume spike and daily uptrend (ADX>25) = long. 
Breakdown below S3 with volume spike and daily downtrend (ADX>25) = short.
Uses daily ADX trend filter to avoid whipsaws in ranging markets. Target: 12-37 trades/year per symbol.
Timeframe: 12h, HTF: 1d
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
    
    # Get daily data for HTF filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate daily ADX(14) for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # first period has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    plus_dm_14 = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values
    minus_dm_14 = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values
    
    # Directional Indicators
    plus_di_14 = 100 * plus_dm_14 / tr_14
    minus_di_14 = 100 * minus_dm_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # Align ADX to 12h timeframe (needs extra delay as ADX is lagging)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx, additional_delay_bars=1)
    strong_trend = adx_aligned > 25
    
    # Calculate Camarilla levels from previous 1d bar
    # Camarilla levels use previous day's OHLC
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close[0] = close_1d[0]  # first bar uses current close as prev close
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    
    # Camarilla R3, R4, S3, S4
    range_1d = prev_high - prev_low
    camarilla_r3 = prev_close + range_1d * 1.1 / 4
    camarilla_s3 = prev_close - range_1d * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume spike detector (20-period volume MA on 12h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period (max of all lookbacks)
    start_idx = 30  # EMA34 needs 30 periods
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(strong_trend[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Trend filter from daily EMA34
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: Price breaks above R3 with volume spike and daily uptrend + strong trend
            if close[i] > camarilla_r3_aligned[i] and volume_spike[i] and uptrend and strong_trend[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 with volume spike and daily downtrend + strong trend
            elif close[i] < camarilla_s3_aligned[i] and volume_spike[i] and downtrend and strong_trend[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Price re-enters below R3 OR daily trend changes to downtrend OR trend weakens
            if close[i] < camarilla_r3_aligned[i] or not uptrend or not strong_trend[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Price re-enters above S3 OR daily trend changes to uptrend OR trend weakens
            if close[i] > camarilla_s3_aligned[i] or not downtrend or not strong_trend[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R3S3_Breakout_1dTrend_ADXFilter"
timeframe = "12h"
leverage = 1.0