#!/usr/bin/env python3
"""
1d_PremiumDiscount_Momentum
Hypothesis: Uses weekly price extremes (52-week high/low) as dynamic support/resistance.
Enters long when price pulls back from weekly high with bullish momentum (price > daily VWAP),
enters short when price bounces from weekly low with bearish momentum (price < daily VWAP).
Volume confirmation filters weak moves. Works in both bull/bear markets by trading mean reversion
within the weekly range while respecting the dominant trend via VWAP.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 52:
        return np.zeros(n)
    
    # === Weekly 52-period high/low for support/resistance ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Rolling 52-week high/low
    high_52w = pd.Series(high_1w).rolling(window=52, min_periods=52).max().values
    low_52w = pd.Series(low_1w).rolling(window=52, min_periods=52).min().values
    
    high_52w_aligned = align_htf_to_ltf(prices, df_1w, high_52w)
    low_52w_aligned = align_htf_to_ltf(prices, df_1w, low_52w)
    
    # === Daily VWAP (volume-weighted average price) ===
    typical_price = (prices['high'].values + prices['low'].values + prices['close'].values) / 3
    vwap_num = np.cumsum(typical_price * prices['volume'].values)
    vwap_den = np.cumsum(prices['volume'].values)
    vwap = np.where(vwap_den != 0, vwap_num / vwap_den, typical_price)
    
    # === Volume confirmation (20-period average) ===
    vol_ma = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(52, n):  # Start after warmup
        # Skip if indicators not ready
        if (np.isnan(high_52w_aligned[i]) or 
            np.isnan(low_52w_aligned[i]) or 
            np.isnan(vwap[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        weekly_high = high_52w_aligned[i]
        weekly_low = low_52w_aligned[i]
        vwap_val = vwap[i]
        vol_ratio_val = vol_ratio[i]
        
        # Avoid division by zero or invalid ranges
        if weekly_high <= weekly_low or weekly_high == 0:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate position within weekly range (0 = at low, 1 = at high)
        range_position = (price_close - weekly_low) / (weekly_high - weekly_low)
        
        if position == 0:
            # Long: near weekly high (overbought) but showing weakness + price below VWAP
            # Actually, we want to buy weakness near high? No - let's reverse logic
            # Long: price near weekly low AND above VWAP (bullish momentum from support)
            # Short: price near weekly high AND below VWAP (bearish momentum from resistance)
            if (range_position < 0.2 and  # Near weekly low (0-20% of range)
                price_close > vwap_val and   # Above VWAP = bullish momentum
                vol_ratio_val > 1.5):      # Strong volume
                signals[i] = 0.25
                position = 1
            elif (range_position > 0.8 and   # Near weekly high (80-100% of range)
                  price_close < vwap_val and   # Below VWAP = bearish momentum
                  vol_ratio_val > 1.5):      # Strong volume
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when price reaches opposite side of range or VWAP crossover
            if position == 1:
                # Exit long if price reaches weekly high OR crosses below VWAP
                if (range_position > 0.8 or price_close < vwap_val):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short if price reaches weekly low OR crosses above VWAP
                if (range_position < 0.2 or price_close > vwap_val):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_PremiumDiscount_Momentum"
timeframe = "1d"
leverage = 1.0