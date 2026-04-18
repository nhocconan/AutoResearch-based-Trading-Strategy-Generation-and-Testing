#!/usr/bin/env python3
"""
12h_1w_LowBollingerBand_Bounce_With_Volume_Confirmation
Hypothesis: Buy near the lower Bollinger Band (20, 2) on weekly timeframe when price shows intraday strength on 12h, confirmed by volume spike. Exit when price touches the upper band or momentum fades. Designed for low frequency (15-30 trades/year) to capture mean-reversion bounces in both bull and bear markets, avoiding whipsaws via weekly trend filter and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for Bollinger Bands
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    sma_20 = pd.Series(close_1w).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_20 = pd.Series(close_1w).rolling(window=bb_period, min_periods=bb_period).std().values
    lower_bb = sma_20 - (bb_std * std_20)
    upper_bb = sma_20 + (bb_std * std_20)
    
    # Align BB to 12h timeframe
    lower_bb_aligned = align_htf_to_ltf(prices, df_1w, lower_bb)
    upper_bb_aligned = align_htf_to_ltf(prices, df_1w, upper_bb)
    sma_20_aligned = align_htf_to_ltf(prices, df_1w, sma_20)
    
    # Weekly trend filter: price above/below 50-week EMA
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 12h volume spike: >1.8x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, 30)  # Warmup for weekly indicators
    
    for i in range(start_idx, n):
        if (np.isnan(lower_bb_aligned[i]) or 
            np.isnan(upper_bb_aligned[i]) or
            np.isnan(sma_20_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        lower_bb_val = lower_bb_aligned[i]
        upper_bb_val = upper_bb_aligned[i]
        sma_20_val = sma_20_aligned[i]
        ema_50_val = ema_50_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price near lower BB, above weekly EMA (uptrend bias), with volume spike
            if price <= lower_bb_val * 1.02 and price > ema_50_val and vol_spike:
                signals[i] = 0.25
                position = 1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price touches upper BB OR loses weekly uptrend
            if price >= upper_bb_val * 0.98 or price < ema_50_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_1w_LowBollingerBand_Bounce_With_Volume_Confirmation"
timeframe = "12h"
leverage = 1.0