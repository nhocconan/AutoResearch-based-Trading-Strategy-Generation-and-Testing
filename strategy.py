#!/usr/bin/env python3
"""
1d_1w_MomentumBreakout_Volume_V1
Concept: Momentum breakout using weekly trend and daily momentum with volume confirmation.
- Long when weekly EMA40 is rising, daily RSI > 55, and price breaks above daily Donchian high(20) with volume > 1.5x average
- Short when weekly EMA40 is falling, daily RSI < 45, and price breaks below daily Donchian low(20) with volume > 1.5x average
- Exit when RSI crosses back to 50 (long) or 50 (short)
- Designed for trending markets (bull/bear) with volume confirmation to avoid false breakouts
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_MomentumBreakout_Volume_V1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # === Calculate weekly EMA40 trend ===
    close_1w = df_1w['close'].values
    ema40_1w = pd.Series(close_1w).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema40_rising = ema40_1w > np.roll(ema40_1w, 1)
    ema40_falling = ema40_1w < np.roll(ema40_1w, 1)
    ema40_rising[0] = False
    ema40_falling[0] = False
    
    # Align weekly EMA trend to daily
    ema40_rising_aligned = align_htf_to_ltf(prices, df_1w, ema40_rising)
    ema40_falling_aligned = align_htf_to_ltf(prices, df_1w, ema40_falling)
    
    # === Daily indicators ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # Neutral when undefined
    
    # Daily Donchian channels (20)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume ratio (current vs 20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(40, 20, 14)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Get values
        ema40_rising_val = ema40_rising_aligned[i]
        ema40_falling_val = ema40_falling_aligned[i]
        rsi_val = rsi[i]
        donchian_high_val = donchian_high[i]
        donchian_low_val = donchian_low[i]
        close_val = close[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema40_rising_val) or np.isnan(ema40_falling_val) or 
            np.isnan(rsi_val) or np.isnan(donchian_high_val) or 
            np.isnan(donchian_low_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Weekly uptrend, RSI > 55, break above Donchian high with volume
            long_condition = (ema40_rising_val and 
                            rsi_val > 55 and 
                            close_val > donchian_high_val and 
                            vol_ratio_val > 1.5)
            
            # Short: Weekly downtrend, RSI < 45, break below Donchian low with volume
            short_condition = (ema40_falling_val and 
                             rsi_val < 45 and 
                             close_val < donchian_low_val and 
                             vol_ratio_val > 1.5)
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI crosses below 50
            if rsi_val < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI crosses above 50
            if rsi_val > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals