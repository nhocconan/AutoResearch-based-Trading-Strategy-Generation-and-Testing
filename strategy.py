#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Bollinger Bands with mean reversion in low volatility regime
# Weekly Bollinger Bands (20,2) define dynamic support/resistance
# Mean reversion when price touches bands with volume confirmation in low volatility (ADX < 25)
# Exit at weekly middle band (20-period SMA)
# Works in bull/bear markets: mean reversion captures pullbacks in trends and range-bound behavior
# Target: 30-100 total trades over 4 years (7-25/year) with 0.30 position sizing

name = "1d_weeklyBB_meanrev_ADXvol_filter_v1"
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
    
    # Calculate weekly Bollinger Bands ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly Bollinger Bands (20, 2)
    weekly_close = df_1w['close'].values
    bb_length = 20
    bb_mult = 2.0
    
    # Basis (SMA)
    basis = pd.Series(weekly_close).rolling(window=bb_length, min_periods=bb_length).mean().values
    # Deviation
    dev = bb_mult * pd.Series(weekly_close).rolling(window=bb_length, min_periods=bb_length).std().values
    # Upper and Lower Bands
    upper = basis + dev
    lower = basis - dev
    
    # Align weekly bands to daily timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1w, upper)
    lower_aligned = align_htf_to_ltf(prices, df_1w, lower)
    basis_aligned = align_htf_to_ltf(prices, df_1w, basis)
    
    # ADX for volatility regime filter (weekly)
    # Calculate True Range
    tr1 = df_1w['high'] - df_1w['low']
    tr2 = np.abs(df_1w['high'] - df_1w['close'].shift(1))
    tr3 = np.abs(df_1w['low'] - df_1w['close'].shift(1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    # Calculate Directional Movement
    dm_plus = np.where((df_1w['high'] - df_1w['high'].shift(1)) > (df_1w['low'].shift(1) - df_1w['low']), 
                       np.maximum(df_1w['high'] - df_1w['high'].shift(1), 0), 0)
    dm_minus = np.where((df_1w['low'].shift(1) - df_1w['low']) > (df_1w['high'] - df_1w['high'].shift(1)), 
                        np.maximum(df_1w['low'].shift(1) - df_1w['low'], 0), 0)
    # Smooth TR, DM+ and DM- with Wilder's smoothing (using EMA with alpha=1/period)
    tr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False).mean().values
    # Calculate DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    # Calculate DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # Align ADX to daily timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Low volatility regime: ADX < 25 (range-bound market)
    vol_regime = adx_aligned < 25
    
    # Session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or np.isnan(basis_aligned[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(vol_regime[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price touches lower band with volume confirmation in low volatility regime
            if close[i] <= lower_aligned[i] and volume_filter[i] and vol_regime[i]:
                signals[i] = 0.30
                position = 1
            # Short: price touches upper band with volume confirmation in low volatility regime
            elif close[i] >= upper_aligned[i] and volume_filter[i] and vol_regime[i]:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price reaches middle band (mean reversion target)
            if close[i] >= basis_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price reaches middle band (mean reversion target)
            if close[i] <= basis_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals