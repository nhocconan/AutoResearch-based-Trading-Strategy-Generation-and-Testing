#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and 1d EMA34 trend filter.
# Long when: price breaks above Donchian high with volume > 1.5x 20-period average and price > 1d EMA34
# Short when: price breaks below Donchian low with volume > 1.5x 20-period average and price < 1d EMA34
# Exit when price returns to Donchian midpoint (mean of high/low) or reverses to opposite side.
# Designed for ~25-50 trades/year per symbol. Works in both bull and bear markets via trend filter.
name = "4h_Donchian20_Breakout_Volume_EMA34Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Donchian channel (20-period) on 4h data
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume average (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Get current levels
        upper = donch_high[i]
        lower = donch_low[i]
        midpoint = donch_mid[i]
        ema_34 = ema_34_1d_aligned[i]
        
        if position == 0:
            # Long breakout: price > upper with volume confirmation and uptrend
            if price > upper and vol > 1.5 * vol_ma and price > ema_34:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price < lower with volume confirmation and downtrend
            elif price < lower and vol > 1.5 * vol_ma and price < ema_34:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to midpoint or breaks below lower (reversal)
            if price <= midpoint or price < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to midpoint or breaks above upper (reversal)
            if price >= midpoint or price > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals