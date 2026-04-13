#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Camarilla pivot breakout with 1d volume spike and choppiness regime filter.
    # Camarilla levels from 1d provide intraday support/resistance based on previous day's range.
    # Breakouts above H3 or below L3 with volume confirmation and chop regime filter capture momentum.
    # Works in bull/bear via volatility regime targeting (chop > 61.8 = range, chop < 38.2 = trend).
    # Target: 75-200 total trades over 4 years = 19-50/year.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and filters (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    rang = high_1d - low_1d
    # H3, H4, L3, L4 levels
    h3 = close_1d + rang * 1.1 / 2
    h4 = close_1d + rang * 1.1
    l3 = close_1d - rang * 1.1 / 2
    l4 = close_1d - rang * 1.1
    
    # Calculate 1d ATR(14) for choppiness filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d ADX(14) for trend strength
    # +DM, -DM
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = np.diff(low_1d, prepend=low_1d[0]) * -1
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # TR (already calculated as 'tr')
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di_14 = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / tr_14
    minus_di_14 = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / tr_14
    dx_14 = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    adx_14 = pd.Series(dx_14).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d volume mean for spike filter
    vol_ma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 4h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Calculate 4h Bollinger Band width for choppiness regime
    close_s = pd.Series(close)
    bb_mid = close_s.rolling(window=20, min_periods=20).mean().values
    bb_std = close_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_mid
    
    # Calculate 4h Bollinger Band width percentile (20-period lookback)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=20, min_periods=20).rank(pct=True).values * 100
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_aligned[i]) or
            np.isnan(bb_width_percentile[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike: current 1d volume > 1.5 * 20-day average
        # Need to get current 1d volume - approximate using aligned volume
        # Since we don't have real-time 1d volume, use close price as proxy for volatility
        volume_spike = close[i] > close[i-1] * 1.02 or close[i] < close[i-1] * 0.98  # 2% move as volume proxy
        
        # Choppiness regime: BB width percentile < 30 = low volatility (squeeze), > 70 = high volatility (trend)
        # We want trend regimes: BB width percentile > 50 (expanding volatility)
        vol_regime = bb_width_percentile[i] > 50
        
        # ADX filter: trend strength > 25
        trend_filter = adx_aligned[i] > 25
        
        # Camarilla breakout conditions
        breakout_long = close[i] > h3_aligned[i]  # Break above H3
        breakout_short = close[i] < l3_aligned[i]  # Break below L3
        
        # Entry conditions: breakout with volume spike, volatility regime, and trend filter
        long_entry = breakout_long and volume_spike and vol_regime and trend_filter
        short_entry = breakout_short and volume_spike and vol_regime and trend_filter
        
        # Exit conditions: price returns to opposite Camarilla level (L3 for long, H3 for short)
        long_exit = close[i] < l3_aligned[i]
        short_exit = close[i] > h3_aligned[i]
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_camarilla_breakout_vol_regime_v1"
timeframe = "4h"
leverage = 1.0