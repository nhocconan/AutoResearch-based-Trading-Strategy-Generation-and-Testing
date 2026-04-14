#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using daily Donchian breakout with 1w ATR volatility filter.
# Long when price breaks above 1d Donchian high (20) with 1w ATR > 1w ATR MA (high volatility regime).
# Short when price breaks below 1d Donchian low (20) with 1w ATR > 1w ATR MA.
# Exit when price returns to 1d Donchian midpoint or volatility drops.
# Designed to capture breakouts in volatile regimes while avoiding choppy low-volatility periods.
# Target: 15-25 trades/year per symbol (60-100 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1d data ONCE for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d Donchian channels (20-period)
    donch_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # Load 1w data ONCE for ATR volatility filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate True Range and ATR on 1w
    tr1 = np.abs(high_1w[1:] - low_1w[1:])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    atr_period = 14
    atr_1w = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    atr_ma_1w = pd.Series(atr_1w).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to lower timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    donch_mid_aligned = align_htf_to_ltf(prices, df_1d, donch_mid)
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    atr_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_ma_1w)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(20, 30)  # Need Donchian and ATR
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or
            np.isnan(donch_mid_aligned[i]) or
            np.isnan(atr_1w_aligned[i]) or
            np.isnan(atr_ma_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when ATR > MA (high volatility regime)
        high_volatility = atr_1w_aligned[i] > atr_ma_1w_aligned[i]
        
        if position == 0:
            # Look for Donchian breakouts in high volatility regime
            # Long: price breaks above Donchian high AND high volatility
            if (close[i] > donch_high_aligned[i] and 
                high_volatility):
                position = 1
                signals[i] = position_size
            # Short: price breaks below Donchian low AND high volatility
            elif (close[i] < donch_low_aligned[i] and 
                  high_volatility):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to Donchian midpoint or volatility drops
            if (close[i] <= donch_mid_aligned[i] or 
                not high_volatility):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to Donchian midpoint or volatility drops
            if (close[i] >= donch_mid_aligned[i] or 
                not high_volatility):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_DonchianBreakout_1wATRFilter_v1"
timeframe = "12h"
leverage = 1.0