#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4x ADX Trend + Volume Spike + Bollinger Band Breakout
# Hypothesis: Strong trends (ADX>25) with volume confirmation and breakouts from
# Bollinger Bands (20,2) capture momentum moves in both bull and bear markets.
# Uses 1d ADX for trend strength filter to avoid whipsaws. Target: 20-50 trades/year.

name = "4h_adx_volume_bb_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 30:
        return np.zeros(n)
    
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # Bollinger Bands (20, 2) on 4h
    bb_period = 20
    bb_std = 2.0
    
    ma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper = ma + bb_std * std
    lower = ma - bb_std * std
    
    # ADX (14) on daily for trend strength
    adx_period = 14
    # True Range
    tr1 = high_daily - low_daily
    tr2 = np.abs(high_daily - np.roll(close_daily, 1))
    tr3 = np.abs(low_daily - np.roll(close_daily, 1))
    tr1[0] = high_daily[0] - low_daily[0]
    tr2[0] = np.abs(high_daily[0] - close_daily[0])
    tr3[0] = np.abs(low_daily[0] - close_daily[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_daily = pd.Series(tr).rolling(window=adx_period, min_periods=adx_period).mean().values
    
    # Directional Movement
    up_move = np.diff(high_daily, prepend=high_daily[0])
    down_move = np.diff(low_daily, prepend=low_daily[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/adx_period, adjust=False).mean().values / atr_daily
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/adx_period, adjust=False).mean().values / atr_daily
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/adx_period, adjust=False).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_daily, adx)
    
    # Volume confirmation: volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=10).mean().values
    vol_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(bb_period, n):
        # Skip if required data not available
        if (np.isnan(ma[i]) or np.isnan(std[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Check volume confirmation
        vol_ok = vol_spike[i]
        
        if position == 1:  # Long position
            # Exit: price closes below middle band or trend weakens
            if close[i] < ma[i] or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price closes above middle band or trend weakens
            if close[i] > ma[i] or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok and adx_aligned[i] > 25:
                # Breakout above upper band with strong trend
                if close[i] > upper[i]:
                    position = 1
                    signals[i] = 0.25
                # Breakdown below lower band with strong trend
                elif close[i] < lower[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals