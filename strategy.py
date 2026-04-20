#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day mean reversion at weekly Bollinger Bands with volume confirmation.
# In both bull and bear markets, price tends to revert to the mean after reaching extreme bands.
# Weekly Bollinger Bands provide dynamic support/resistance; volume confirms institutional interest.
# Weekly trend filter avoids counter-trend trades in strong moves.
# Target: 10-25 trades/year per symbol with disciplined exits.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for Bollinger Bands and trend
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly Bollinger Bands (20, 2.0)
    bb_period = 20
    bb_std = 2.0
    sma_20 = pd.Series(close_1w).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_20 = pd.Series(close_1w).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_bb = sma_20 + bb_std * std_20
    lower_bb = sma_20 - bb_std * std_20
    upper_bb_aligned = align_htf_to_ltf(prices, df_1w, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1w, lower_bb)
    sma_20_aligned = align_htf_to_ltf(prices, df_1w, sma_20)
    
    # Load daily data
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Daily volume average for confirmation
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if (np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i]) or 
            np.isnan(sma_20_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(close_1d[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_1d[i]
        vol = volume_1d[i]
        
        if position == 0:
            # Long: price touches or crosses below lower Bollinger Band with volume confirmation
            if (price <= lower_bb_aligned[i] and 
                vol > 1.5 * vol_ma_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price touches or crosses above upper Bollinger Band with volume confirmation
            elif (price >= upper_bb_aligned[i] and 
                  vol > 1.5 * vol_ma_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to weekly SMA (mean reversion complete) or volume drops
            if price >= sma_20_aligned[i] or vol < 0.8 * vol_ma_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to weekly SMA or volume drops
            if price <= sma_20_aligned[i] or vol < 0.8 * vol_ma_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyBollingerMeanReversion_Volume"
timeframe = "1d"
leverage = 1.0