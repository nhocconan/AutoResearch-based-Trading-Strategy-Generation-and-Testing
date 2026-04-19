#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and Choppiness regime filter
# - 4h Donchian(20) upper/lower bands define breakout levels
# - 1d volume > 1.5x 20-period average for conviction
# - 1d Choppiness Index > 61.8 for range regime (mean reversion at bands), < 38.2 for trend (follow breakout)
# - In trend regime (CHOP < 38.2): long on upper band breakout, short on lower band breakdown
# - In range regime (CHOP > 61.8): long at lower band, short at upper band (mean reversion)
# - Exit on opposite band touch or regime change
# - Position size: 0.25 (25%) to manage drawdown
# - Designed to work in both bull and bear markets by adapting to regime
# - Target: 20-50 trades/year to avoid excessive fee drag

name = "4h_Donchian20_1dVolume_Chop_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume and Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    
    # 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # 1d Choppiness Index (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with close_1d
    
    # ATR(14)
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # +DM and -DM
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed +DM and -DM
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # +DI and -DI
    plus_di = 100 * plus_dm_smooth / atr_1d
    minus_di = 100 * minus_dm_smooth / atr_1d
    
    # DX and Choppiness Index
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    chop = 100 * np.log10(pd.Series(atr_1d).rolling(window=14, min_periods=14).sum() / 
                          (np.abs(high_1d - low_1d).rolling(window=14, min_periods=14).sum())) / np.log10(14)
    chop = chop.values
    
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # 4h Donchian channels (20-period)
    donch_h = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_l = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_h[i]) or np.isnan(donch_l[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 1d volume > 1.5x average
        volume_filter = vol_ma_1d_aligned[i] > 0 and volume[i] > 1.5 * vol_ma_1d_aligned[i]
        
        if not volume_filter:
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
            
        if position == 0:
            # Determine regime: chop > 61.8 = range, chop < 38.2 = trend
            if chop_aligned[i] > 61.8:  # Range regime - mean reversion
                # Long at lower band, short at upper band
                if close[i] <= donch_l[i]:
                    signals[i] = 0.25
                    position = 1
                elif close[i] >= donch_h[i]:
                    signals[i] = -0.25
                    position = -1
            elif chop_aligned[i] < 38.2:  # Trend regime - follow breakout
                # Long on upper band breakout, short on lower band breakdown
                if close[i] > donch_h[i]:
                    signals[i] = 0.25
                    position = 1
                elif close[i] < donch_l[i]:
                    signals[i] = -0.25
                    position = -1
            else:  # Neutral regime - no action
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit at opposite band or regime change to strong trend against position
            if chop_aligned[i] < 38.2 and close[i] < donch_l[i]:  # Trend regime break down
                signals[i] = 0.0
                position = 0
            elif chop_aligned[i] > 61.8 and close[i] >= donch_h[i]:  # Range regime reached upper band
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit at opposite band or regime change to strong trend against position
            if chop_aligned[i] < 38.2 and close[i] > donch_h[i]:  # Trend regime break up
                signals[i] = 0.0
                position = 0
            elif chop_aligned[i] > 61.8 and close[i] <= donch_l[i]:  # Range regime reached lower band
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals