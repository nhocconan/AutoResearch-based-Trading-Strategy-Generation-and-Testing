# USDC/USDT Exchange Rate Monitoring Strategy
# Hypothesis: USDC/USDT is a stablecoin pair that should maintain a tight 1:1 peg.
# Deviations from parity indicate market stress or arbitrage opportunities.
# This strategy monitors the USDC/USDT price on Binance and takes small positions
# when the price deviates significantly from 1.0, expecting mean reversion.
# Works in both bull and bear markets as it's based on mean reversion of a stable peg.
# Very low frequency - only triggers during significant de-pegging events.

name = "USDC_USDT_Peg_Monitor"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Calculate deviation from 1.0 peg
    peg_deviation = close - 1.0
    
    # Calculate rolling statistics for z-score (20-period window)
    peg_mean = pd.Series(peg_deviation).rolling(window=20, min_periods=20).mean().values
    peg_std = pd.Series(peg_deviation).rolling(window=20, min_periods=20).std().values
    
    # Calculate z-score of peg deviation
    z_score = np.zeros_like(peg_deviation)
    for i in range(len(z_score)):
        if peg_std[i] > 0:
            z_score[i] = (peg_deviation[i] - peg_mean[i]) / peg_std[i]
        else:
            z_score[i] = 0
    
    # Get 1-day data for additional confirmation (market stress indicator)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Daily volatility as market stress indicator
    daily_returns = np.abs(np.diff(df_1d['close'].values) / df_1d['close'].values[:-1])
    daily_vol = pd.Series(daily_returns).rolling(window=10, min_periods=10).mean().values
    daily_vol_aligned = align_htf_to_ltf(prices, df_1d, daily_vol)
    
    # Volume confirmation - unusually high volume during de-pegging
    volume_ma = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long (USDC undervalued), -1: short (USDC overvalued)
    
    # Only trade during significant market stress to avoid noise
    volatility_threshold = 0.02  # 2% daily volatility threshold
    
    for i in range(20, n):
        # Skip if any critical value is NaN
        if (np.isnan(z_score[i]) or np.isnan(daily_vol_aligned[i]) or 
            np.isnan(volume_ma[i]) or volume_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Only trade when market volatility is elevated (stress conditions)
        if daily_vol_aligned[i] < volatility_threshold:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: USDC significantly undervalued (z-score < -2) with volume confirmation
            if z_score[i] < -2.0 and prices['volume'].values[i] > volume_ma[i]:
                signals[i] = 0.25  # 25% position
                position = 1
            # Short: USDC significantly overvalued (z-score > 2) with volume confirmation
            elif z_score[i] > 2.0 and prices['volume'].values[i] > volume_ma[i]:
                signals[i] = -0.25  # 25% position
                position = -1
        elif position == 1:
            # Exit long: USDC returns to peg (z-score > -0.5)
            if z_score[i] > -0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: USDC returns to peg (z-score < 0.5)
            if z_score[i] < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3