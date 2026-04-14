#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d timeframe with 1w ATR-based volatility filter and 1w Donchian breakout
# The strategy identifies low volatility periods using 1w ATR (normalized by price) and
# trades breakouts from the 1w Donchian channel (20 periods) in the direction of the breakout.
# Volatility filter reduces false breakouts during high-noise periods, improving robustness
# in both bull and bear markets by focusing on volatility expansion after contraction.
# Uses only 1w data for both filter and signal to avoid overtrading and ensure alignment.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1w data ONCE for ATR and Donchian channels
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w ATR (14 periods) for volatility measurement
    atr_length = 14
    tr1 = df_1w['high'] - df_1w['low']
    tr2 = abs(df_1w['high'] - df_1w['close'].shift(1))
    tr3 = abs(df_1w['low'] - df_1w['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=atr_length, min_periods=atr_length).mean().values
    
    # Normalize ATR by price to make it scale-invariant (ATR as % of price)
    price_for_atr = df_1w['close'].values
    atr_pct = np.where(price_for_atr != 0, atr / price_for_atr, 0)
    
    # Calculate 1w Donchian channels (20 periods)
    donch_length = 20
    donch_high = pd.Series(df_1w['high']).rolling(window=donch_length, min_periods=donch_length).max().values
    donch_low = pd.Series(df_1w['low']).rolling(window=donch_length, min_periods=donch_length).min().values
    
    # Align ATR% and Donchian channels to 1d timeframe
    atr_pct_aligned = align_htf_to_ltf(prices, df_1w, atr_pct)
    donch_high_aligned = align_htf_to_ltf(prices, df_1w, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1w, donch_low)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, donch_length)  # Need enough for ATR and Donchian
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(atr_pct_aligned[i]) or 
            np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Volatility filter: Low volatility regime (ATR% below 30th percentile of last 50 periods)
        # We calculate percentile rank manually to avoid look-ahead
        if i >= 50:
            atr_slice = atr_pct_aligned[max(0, i-50):i]
            if len(atr_slice) > 0:
                # Remove NaN values for percentile calculation
                atr_slice_clean = atr_slice[~np.isnan(atr_slice)]
                if len(atr_slice_clean) > 0:
                    sorted_atr = np.sort(atr_slice_clean)
                    current_atr = atr_pct_aligned[i]
                    # Count how many values are less than current ATR%
                    rank = np.searchsorted(sorted_atr, current_atr, side='left')
                    percentile = (rank / len(sorted_atr)) * 100
                    low_vol = percentile <= 30  # Low volatility regime
                else:
                    low_vol = False
            else:
                low_vol = False
        else:
            low_vol = False
        
        # Breakout signals from 1w Donchian
        breakout_up = price > donch_high_aligned[i]
        breakout_down = price < donch_low_aligned[i]
        
        if position == 0:
            # Enter long: low volatility + upward breakout
            if low_vol and breakout_up:
                position = 1
                signals[i] = position_size
            # Enter short: low volatility + downward breakout
            elif low_vol and breakout_down:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Donchian low OR volatility increases (ATR% > 70th percentile)
            if i >= 50:
                atr_slice = atr_pct_aligned[max(0, i-50):i]
                if len(atr_slice) > 0:
                    atr_slice_clean = atr_slice[~np.isnan(atr_slice)]
                    if len(atr_slice_clean) > 0:
                        sorted_atr = np.sort(atr_slice_clean)
                        current_atr = atr_pct_aligned[i]
                        rank = np.searchsorted(sorted_atr, current_atr, side='left')
                        percentile = (rank / len(sorted_atr)) * 100
                        high_vol = percentile >= 70
                    else:
                        high_vol = False
                else:
                    high_vol = False
            else:
                high_vol = False
            
            if price < donch_low_aligned[i] or high_vol:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above Donchian high OR volatility increases
            if i >= 50:
                atr_slice = atr_pct_aligned[max(0, i-50):i]
                if len(atr_slice) > 0:
                    atr_slice_clean = atr_slice[~np.isnan(atr_slice)]
                    if len(atr_slice_clean) > 0:
                        sorted_atr = np.sort(atr_slice_clean)
                        current_atr = atr_pct_aligned[i]
                        rank = np.searchsorted(sorted_atr, current_atr, side='left')
                        percentile = (rank / len(sorted_atr)) * 100
                        high_vol = percentile >= 70
                    else:
                        high_vol = False
                else:
                    high_vol = False
            else:
                high_vol = False
            
            if price > donch_high_aligned[i] or high_vol:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1wATR_Percentile_1wDonchian_Breakout_v1"
timeframe = "1d"
leverage = 1.0