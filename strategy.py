#!/usr/bin/env python3
# Hypothesis: 6h Bollinger Band Squeeze Breakout with 1d ADX trend filter and volume confirmation.
# Long when: BB width < 20th percentile (squeeze), close breaks above upper band, 1d ADX > 25 (trending), volume > 1.5x 20-bar average.
# Short when: BB width < 20th percentile (squeeze), close breaks below lower band, 1d ADX > 25 (trending), volume > 1.5x 20-bar average.
# Exit when: price re-enters the Bollinger Bands (mean reversion of squeeze) OR BB width expands above 50th percentile (squeeze end).
# Uses 1d HTF for ADX trend filter to avoid whipsaws in ranging markets. Bollinger squeeze identifies low volatility periods primed for breakout.
# Volume confirmation ensures breakout has participation. Target: 50-150 total trades over 4 years (12-37/year) to stay within fee drag limits for 6h timeframe.
# Bollinger Bands effectively capture volatility contraction/expansion cycles that precede significant moves in both bull and bear markets.

name = "6h_BollingerSqueeze_Breakout_1dADX_6hVolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 6h Indicators (LTF) ---
    # Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_stddev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = sma + (bb_std * bb_stddev)
    lower_band = sma - (bb_std * bb_stddev)
    bb_width = (upper_band - lower_band) / sma  # normalized width
    
    # BB width percentiles for squeeze detection (using 50-bar lookback)
    bb_width_series = pd.Series(bb_width)
    bb_width_pct = bb_width_series.rolling(window=50, min_periods=50).rank(pct=True).values * 100
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d ADX(14) - trend filter
    # Calculate True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with original index
    
    # Calculate Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smooth TR and DM
    tr_period = 14
    atr = pd.Series(tr).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after BB width percentile lookback
        # Skip if missing data
        if (np.isnan(sma[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or
            np.isnan(bb_width_pct[i]) or np.isnan(volume_confirm[i]) or
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: BB squeeze (width < 20th percentile) AND breakout above upper band AND trending (ADX > 25) AND volume confirm
            if (bb_width_pct[i] < 20 and 
                close[i] > upper_band[i] and 
                adx_aligned[i] > 25 and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: BB squeeze (width < 20th percentile) AND breakout below lower band AND trending (ADX > 25) AND volume confirm
            elif (bb_width_pct[i] < 20 and 
                  close[i] < lower_band[i] and 
                  adx_aligned[i] > 25 and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price re-enters BB (mean reversion) OR squeeze ends (width > 50th percentile)
            if (close[i] < sma[i] or  # re-enter below middle band
                bb_width_pct[i] > 50):  # squeeze ended
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price re-enters BB (mean reversion) OR squeeze ends (width > 50th percentile)
            if (close[i] > sma[i] or  # re-enter above middle band
                bb_width_pct[i] > 50):  # squeeze ended
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals