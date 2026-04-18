#!/usr/bin/env python3
"""
12h_1w_LowBollingerBand_Bounce_With_Volume_Confirmation
Hypothesis: Buy when price touches the lower Bollinger Band (20,2) on 12h and weekly Bollinger Band width is at a 3-month low, with volume confirmation. Sell when price reaches the middle band or upper band. Designed for low frequency (15-30 trades/year) to capture mean-reversion bounces in ranging markets while avoiding whipsaws in strong trends via volatility regime filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data (same as primary) and 1w data for regime filter
    df_12h = get_htf_data(prices, '12h')
    df_1w = get_htf_data(prices, '1w')
    
    # 12h Bollinger Bands (20,2)
    close_12h = df_12h['close'].values
    ma_20 = pd.Series(close_12h).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_12h).rolling(window=20, min_periods=20).std().values
    lower_bb = ma_20 - 2 * std_20
    middle_bb = ma_20
    upper_bb = ma_20 + 2 * std_20
    lower_bb_aligned = align_htf_to_ltf(prices, df_12h, lower_bb)
    middle_bb_aligned = align_htf_to_ltf(prices, df_12h, middle_bb)
    upper_bb_aligned = align_htf_to_ltf(prices, df_12h, upper_bb)
    
    # 1w Bollinger Band Width for regime filter (low volatility = ranging)
    close_1w = df_1w['close'].values
    ma_20_1w = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    std_20_1w = pd.Series(close_1w).rolling(window=20, min_periods=20).std().values
    upper_bb_1w = ma_20_1w + 2 * std_20_1w
    lower_bb_1w = ma_20_1w - 2 * std_20_1w
    bb_width_1w = (upper_bb_1w - lower_bb_1w) / ma_20_1w  # normalized width
    
    # 3-month (approximately 12 weekly bars) lowest bb width
    bb_width_min_12w = pd.Series(bb_width_1w).rolling(window=12, min_periods=12).min().values
    low_volatility_regime = bb_width_1w <= bb_width_min_12w * 1.1  # within 10% of minimum
    low_volatility_aligned = align_htf_to_ltf(prices, df_1w, low_volatility_regime)
    
    # Volume spike: >1.8x 30-period average on 12h
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(35, 30)  # Warmup for BB and volume
    
    for i in range(start_idx, n):
        if (np.isnan(lower_bb_aligned[i]) or 
            np.isnan(middle_bb_aligned[i]) or
            np.isnan(upper_bb_aligned[i]) or
            np.isnan(low_volatility_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        lower = lower_bb_aligned[i]
        middle = middle_bb_aligned[i]
        upper = upper_bb_aligned[i]
        low_vol = low_volatility_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price at or below lower BB, low volatility regime, volume spike
            if price <= lower and low_vol and vol_spike:
                signals[i] = 0.25
                position = 1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price touches or crosses middle band, or volatility breaks out
            if price >= middle or not low_vol:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_1w_LowBollingerBand_Bounce_With_Volume_Confirmation"
timeframe = "12h"
leverage = 1.0