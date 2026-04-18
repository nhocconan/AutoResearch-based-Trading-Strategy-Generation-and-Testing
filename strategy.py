#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d price closing above/below weekly Bollinger Bands with volume confirmation and ADX trend filter.
# Long when close > upper BB(20,2) on weekly, volume > 1.5x 20-period average, and weekly ADX > 25.
# Short when close < lower BB(20,2) on weekly, volume > 1.5x 20-period average, and weekly ADX > 25.
# Exit when price returns to weekly middle band (20-period SMA).
# Uses weekly Bollinger Bands for dynamic support/resistance, volume surge for conviction, ADX for trend strength.
# Designed for ~10-20 trades/year per symbol to minimize fee drag in ranging markets.
name = "1d_WeeklyBB_Width_ADX_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for Bollinger Bands and ADX
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly OHLC
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    sma_20 = pd.Series(close_1w).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_20 = pd.Series(close_1w).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_bb = sma_20 + (bb_std * std_20)
    lower_bb = sma_20 - (bb_std * std_20)
    middle_bb = sma_20  # 20-period SMA for exit
    
    # Weekly ADX (14-period)
    adx_period = 14
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w), 
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)), 
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_period = pd.Series(tr).rolling(window=adx_period, min_periods=adx_period).sum().values
    dm_plus_period = pd.Series(dm_plus).rolling(window=adx_period, min_periods=adx_period).sum().values
    dm_minus_period = pd.Series(dm_minus).rolling(window=adx_period, min_periods=adx_period).sum().values
    
    # DI+ and DI-
    di_plus = 100 * (dm_plus_period / tr_period)
    di_minus = 100 * (dm_minus_period / tr_period)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=adx_period, min_periods=adx_period).mean().values
    
    # Align weekly indicators to daily timeframe
    upper_bb_aligned = align_htf_to_ltf(prices, df_1w, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1w, lower_bb)
    middle_bb_aligned = align_htf_to_ltf(prices, df_1w, middle_bb)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i]) or 
            np.isnan(middle_bb_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        upper_bb_val = upper_bb_aligned[i]
        lower_bb_val = lower_bb_aligned[i]
        middle_bb_val = middle_bb_aligned[i]
        adx_val = adx_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Long: close above upper BB with volume surge and strong trend (ADX > 25)
            if close_val > upper_bb_val and vol_filter and adx_val > 25:
                signals[i] = 0.25
                position = 1
            # Short: close below lower BB with volume surge and strong trend (ADX > 25)
            elif close_val < lower_bb_val and vol_filter and adx_val > 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to middle BB
            if close_val <= middle_bb_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to middle BB
            if close_val >= middle_bb_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals