#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data for volatility regime and price channels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Bollinger Bands width for volatility regime (1d)
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Bollinger Bands (20, 2)
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    bb_width = (upper_bb - lower_bb) / sma_20
    
    # Bollinger Bandwidth percentile (252-day lookback for regime)
    bb_width_series = pd.Series(bb_width)
    bb_percentile = bb_width_series.rolling(window=252, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Donchian channels (20-period) for breakout signals
    highest_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align 1d indicators to 4h timeframe
    bb_percentile_aligned = align_htf_to_ltf(prices, df_1d, bb_percentile)
    highest_20_aligned = align_htf_to_ltf(prices, df_1d, highest_20)
    lowest_20_aligned = align_htf_to_ltf(prices, df_1d, lowest_20)
    
    # 4h volume confirmation (20-period average)
    vol_ma_4h = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Price arrays
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup
        # Skip if data not ready
        if (np.isclose(bb_percentile_aligned[i], 0) or 
            np.isnan(highest_20_aligned[i]) or 
            np.isnan(lowest_20_aligned[i]) or 
            np.isnan(vol_ma_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        bb_percentile_val = bb_percentile_aligned[i]
        upper_channel = highest_20_aligned[i]
        lower_channel = lowest_20_aligned[i]
        vol_ma = vol_ma_4h[i]
        vol = volume[i]
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirm = vol > 1.3 * vol_ma
        
        # Regime filter: trending when BB percentile < 30 (low volatility breakout favorable)
        trending_regime = bb_percentile_val < 30
        
        if position == 0:
            # Long: Breakout above upper Donchian channel + volume + trending regime
            if close[i] > upper_channel and volume_confirm and trending_regime:
                signals[i] = 0.25
                position = 1
            # Short: Breakout below lower Donchian channel + volume + trending regime
            elif close[i] < lower_channel and volume_confirm and trending_regime:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: Opposite breakout or volatility expansion (regime change)
            exit_signal = False
            
            if position == 1:  # long position
                # Exit on breakdown below lower channel or volatility expansion
                if close[i] < lower_channel or bb_percentile_val > 70:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit on breakout above upper channel or volatility expansion
                if close[i] > upper_channel or bb_percentile_val > 70:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_BBWidthRegime_Volume"
timeframe = "4h"
leverage = 1.0