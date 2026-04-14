#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter with 1d Donchian breakout
# In choppy markets (CHOP > 61.8): mean-revert at 1d Donchian bands
# In trending markets (CHOP < 38.2): breakout in direction of 1d trend
# Uses Choppiness Index for regime detection, Donchian for entry/exit
# Target: 75-200 total trades over 4 years (19-50/year) to balance opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1d data ONCE before loop for trend and Donchian
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 14-period Choppiness Index on 4h
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr.rolling(window=14, min_periods=14).sum()
    max_hh = pd.Series(high).rolling(window=14, min_periods=14).max()
    min_ll = pd.Series(low).rolling(window=14, min_periods=14).min()
    chop = 100 * np.log10(atr14 / (max_hh - min_ll)) / np.log10(14)
    
    # Calculate 1d Donchian channels (20-period)
    high_1d = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max()
    low_1d = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min()
    donchian_1d_upper = high_1d
    donchian_1d_lower = low_1d
    donchian_1d_mid = (donchian_1d_upper + donchian_1d_lower) / 2
    
    # Calculate 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean()
    
    # Align 1d indicators to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop.values)  # Chop is 1d indicator
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_1d_upper.values)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_1d_lower.values)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_1d_mid.values)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d.values)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(chop_aligned[i]) or 
            np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or 
            np.isnan(ema_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        chop_val = chop_aligned[i]
        donchian_upper = donchian_upper_aligned[i]
        donchian_lower = donchian_lower_aligned[i]
        donchian_mid = donchian_mid_aligned[i]
        ema_50 = ema_50_aligned[i]
        price = close[i]
        
        if position == 0:
            # Determine market regime
            is_choppy = chop_val > 61.8
            is_trending = chop_val < 38.2
            
            if is_choppy:
                # Mean reversion in chop: fade at Donchian bands
                if price <= donchian_lower:
                    position = 1  # Long at lower band
                    signals[i] = position_size
                elif price >= donchian_upper:
                    position = -1  # Short at upper band
                    signals[i] = -position_size
                else:
                    signals[i] = 0.0
            elif is_trending:
                # Trend following: breakout in direction of 1d EMA
                if price > donchian_upper and ema_50 > donchian_mid:
                    position = 1  # Long breakout in uptrend
                    signals[i] = position_size
                elif price < donchian_lower and ema_50 < donchian_mid:
                    position = -1  # Short breakout in downtrend
                    signals[i] = -position_size
                else:
                    signals[i] = 0.0
            else:
                # Neutral chop (38.2 <= CHOP <= 61.8): no trade
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses Donchian midline OR opposite signal in same regime
            if price < donchian_mid:
                position = 0
                signals[i] = 0.0
            elif chop_val > 61.8 and price >= donchian_upper:  # Reverse in chop
                position = -1
                signals[i] = -position_size
            elif chop_val < 38.2 and price < donchian_lower and ema_50 < donchian_mid:  # Reverse in trend
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses Donchian midline OR opposite signal in same regime
            if price > donchian_mid:
                position = 0
                signals[i] = 0.0
            elif chop_val > 61.8 and price <= donchian_lower:  # Reverse in chop
                position = 1
                signals[i] = position_size
            elif chop_val < 38.2 and price > donchian_upper and ema_50 > donchian_mid:  # Reverse in trend
                position = 1
                signals[i] = position_size
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Chop_Donchian_1dEMA"
timeframe = "4h"
leverage = 1.0