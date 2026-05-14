#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversion with 1d trend filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions. In ranging markets (CHOP > 50),
# we fade extremes: long when %R < -80 and short when %R > -20. In trending markets
# (CHOP < 50), we follow the 1d EMA34 trend. Volume spike confirms momentum. Designed
# to work in both bull and bear markets by adapting to regime.

name = "4h_WilliamsR_MeanReversion_1dTrend_VolumeSpike_Regime"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and CHOP regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 trend filter
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate 1d CHOP regime: CHOP > 50 = ranging (mean revert), CHOP < 50 = trending (trend follow)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    tr_1d = np.maximum(high_1d - low_1d, np.maximum(np.abs(high_1d - np.roll(close_1d_vals, 1)), np.abs(low_1d - np.roll(close_1d_vals, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]  # first TR
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    highest_high_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denom = highest_high_1d - lowest_low_1d
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)  # avoid division by zero
    chop_1d = 100 * (np.log10(atr_1d * np.sqrt(14) / chop_denom) / np.log10(10))
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate Williams %R on 4h: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_4h = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_4h = pd.Series(low).rolling(window=14, min_periods=14).min().values
    wr_denom = highest_high_4h - lowest_low_4h
    wr_denom = np.where(wr_denom == 0, 1e-10, wr_denom)  # avoid division by zero
    williams_r = ((highest_high_4h - close) / wr_denom) * -100
    
    # Calculate volume regime: current 4h volume > 1.8x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get current values
        wr_val = williams_r[i]
        ema_trend = ema_34_aligned[i]
        chop_val = chop_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        # Skip if any value is NaN
        if np.isnan(wr_val) or np.isnan(ema_trend) or np.isnan(chop_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Regime-based entry conditions
        if chop_val > 50:  # Ranging market: mean reversion
            # Long: oversold (%R < -80) with volume spike
            long_entry = (wr_val < -80) and vol_spike
            # Short: overbought (%R > -20) with volume spike
            short_entry = (wr_val > -20) and vol_spike
        else:  # Trending market: follow 1d EMA34 trend
            # Long: above EMA34 with volume spike
            long_entry = (close[i] > ema_trend) and vol_spike
            # Short: below EMA34 with volume spike
            short_entry = (close[i] < ema_trend) and vol_spike
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit on close below EMA34 (trend change) or Wolfe wave completion
            if close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit on close above EMA34 (trend change)
            if close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals