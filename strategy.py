#!/usr/bin/env python3
"""
1d_WickReversal_VolumeSpike_v1
Concept: Daily reversal at extreme wicks with volume confirmation and weekly trend filter.
- Long: Low touches or breaks below 1w Bollinger lower band AND close > open (bullish candle) AND volume spike
- Short: High touches or breaks above 1w Bollinger upper band AND close < open (bearish candle) AND volume spike
- Exit: Opposite signal triggers or price crosses 1w Bollinger middle band
- Uses 1w Bollinger Bands (20, 2) for trend context and dynamic levels
- Volume spike: daily volume > 2.0x 20-period average
- Position sizing: 0.25
- Target: 30-80 total trades over 4 years (7-20/year)
- Works in bull/bear: 1w Bollinger adapts to volatility, wick rejection shows exhaustion
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WickReversal_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # === Weekly: Bollinger Bands (20, 2) ===
    close_1w = df_1w['close'].values
    bb_length = 20
    bb_mult = 2.0
    
    basis = pd.Series(close_1w).rolling(window=bb_length, min_periods=bb_length).mean().values
    dev = bb_mult * pd.Series(close_1w).rolling(window=bb_length, min_periods=bb_length).std().values
    upper_band = basis + dev
    lower_band = basis - dev
    
    # Align to daily timeframe
    bb_upper_aligned = align_htf_to_ltf(prices, df_1w, upper_band)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1w, lower_band)
    bb_middle_aligned = align_htf_to_ltf(prices, df_1w, basis)
    
    # === Daily: Volume Spike Filter ===
    volume_1d = df_1w['volume'].values  # Weekly volume for confirmation
    vol_ma_20_1w = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need at least 20 weeks for Bollinger Bands
    
    for i in range(start_idx, n):
        # Get values
        bb_upper = bb_upper_aligned[i]
        bb_lower = bb_lower_aligned[i]
        bb_middle = bb_middle_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(bb_upper) or np.isnan(bb_lower) or np.isnan(bb_middle)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Price action
        open_price = prices['open'].iloc[i]
        high_price = prices['high'].iloc[i]
        low_price = prices['low'].iloc[i]
        close_price = prices['close'].iloc[i]
        
        # Weekly volume condition: current weekly volume > 2.0x 20-period average
        # Note: Using weekly volume aligned to daily
        vol_1w_vals = df_1w['volume'].values
        vol_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_1w_vals)
        current_vol = vol_1w_aligned[i]
        vol_condition = current_vol > 2.0 * vol_ma_20_1w_aligned[i]
        
        # Wick conditions
        # Bullish rejection: long lower wick touching/below BB lower with bullish close
        lower_wick_touch = low_price <= bb_lower
        bullish_candle = close_price > open_price
        bullish_wick_reject = lower_wick_touch and bullish_candle
        
        # Bearish rejection: long upper wick touching/above BB upper with bearish close
        upper_wick_touch = high_price >= bb_upper
        bearish_candle = close_price < open_price
        bearish_wick_reject = upper_wick_touch and bearish_candle
        
        if position == 0:
            # Long: bullish wick rejection at lower band with volume spike
            if bullish_wick_reject and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: bearish wick rejection at upper band with volume spike
            elif bearish_wick_reject and vol_condition:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: bearish reversal signal or price crosses above middle band
            if bearish_wick_reject and vol_condition:
                signals[i] = 0.0
                position = 0
            elif close_price > bb_middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: bullish reversal signal or price crosses below middle band
            if bullish_wick_reject and vol_condition:
                signals[i] = 0.0
                position = 0
            elif close_price < bb_middle:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals