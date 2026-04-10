#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout + 1d ATR volatility filter + chop regime
# - Primary: 4h price breaks above/below 20-period Donchian channel
# - HTF: 1d ATR(14) > 1.5x its 50-period MA (high volatility regime)
# - Regime filter: 4h Choppiness Index (14) > 61.8 = ranging market (mean reversion)
# - Long: Price breaks above Donchian HIGH + volatility filter + chop ranging
# - Short: Price breaks below Donchian LOW + volatility filter + chop ranging
# - Exit: Price returns to Donchian midpoint (mean reversion to equilibrium)
# - Position sizing: 0.25 (discrete level, balances return/drawdown, reduces fee churn)
# - Works in bull/bear: Donchian adapts to volatility, volatility filter avoids low-vol whipsaws, chop filter targets ranging markets
# - Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe

name = "4h_1d_donchian_vol_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 55:
        return np.zeros(n)
    
    # Pre-compute 4h data
    close_4h = prices['close'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 4h Donchian channel (20)
    donchian_high = np.full(len(close_4h), np.nan)
    donchian_low = np.full(len(close_4h), np.nan)
    donchian_mid = np.full(len(close_4h), np.nan)
    
    for i in range(19, len(close_4h)):
        if not (np.isnan(high_4h[i-19:i+1]).any() or np.isnan(low_4h[i-19:i+1]).any()):
            donchian_high[i] = np.max(high_4h[i-19:i+1])
            donchian_low[i] = np.min(low_4h[i-19:i+1])
            donchian_mid[i] = (donchian_high[i] + donchian_low[i]) / 2
    
    # Calculate 1d ATR(14)
    atr_1d = np.full(len(close_1d), np.nan)
    tr_1d = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):
        if not (np.isnan(high_1d[i]) or np.isnan(low_1d[i]) or np.isnan(close_1d[i-1])):
            tr_1d[i] = max(
                high_1d[i] - low_1d[i],
                abs(high_1d[i] - close_1d[i-1]),
                abs(low_1d[i] - close_1d[i-1])
            )
    
    for i in range(13, len(tr_1d)):
        if not np.isnan(tr_1d[i-13:i+1]).any():
            atr_1d[i] = np.mean(tr_1d[i-13:i+1])
    
    # Calculate 1d ATR MA(50)
    atr_ma_50_1d = np.full(len(atr_1d), np.nan)
    for i in range(49, len(atr_1d)):
        if not np.isnan(atr_1d[i-49:i+1]).any():
            atr_ma_50_1d[i] = np.mean(atr_1d[i-49:i+1])
    
    # Calculate 4h Choppiness Index (14)
    chop = np.full(len(close_4h), np.nan)
    
    # True Range for Chop
    tr_4h = np.full(len(close_4h), np.nan)
    for i in range(1, len(close_4h)):
        if not (np.isnan(high_4h[i]) or np.isnan(low_4h[i]) or np.isnan(close_4h[i-1])):
            tr_4h[i] = max(
                high_4h[i] - low_4h[i],
                abs(high_4h[i] - close_4h[i-1]),
                abs(low_4h[i] - close_4h[i-1])
            )
    
    # ATR sum for Chop denominator
    atr_sum_4h = np.full(len(tr_4h), np.nan)
    for i in range(13, len(tr_4h)):
        if not np.isnan(tr_4h[i-13:i+1]).any():
            atr_sum_4h[i] = np.sum(tr_4h[i-13:i+1])
    
    # Choppiness Index
    for i in range(13, len(close_4h)):
        if not (np.isnan(high_4h[i]) or np.isnan(low_4h[i]) or np.isnan(atr_sum_4h[i])):
            highest_high = np.max(high_4h[i-13:i+1])
            lowest_low = np.min(low_4h[i-13:i+1])
            if atr_sum_4h[i] > 0 and (highest_high - lowest_low) > 0:
                chop[i] = 100 * np.log10(atr_sum_4h[i] / (highest_high - lowest_low)) / np.log10(14)
    
    # Align HTF indicators to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, prices, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, prices, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, prices, donchian_mid)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    atr_ma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_50_1d)
    chop_aligned = align_htf_to_ltf(prices, prices, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(atr_ma_50_1d_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: 1d ATR > 1.5x its 50-period MA
        vol_filter = atr_1d_aligned[i] > 1.5 * atr_ma_50_1d_aligned[i]
        
        # Chop regime filter: CHOP > 61.8 = ranging market (mean reversion)
        chop_ranging = chop_aligned[i] > 61.8
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above Donchian HIGH + volatility filter + chop ranging
            if close_4h[i] > donchian_high_aligned[i] and vol_filter and chop_ranging:
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below Donchian LOW + volatility filter + chop ranging
            elif close_4h[i] < donchian_low_aligned[i] and vol_filter and chop_ranging:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Price returns to Donchian midpoint (mean reversion to equilibrium)
            if position == 1:  # Long position
                if close_4h[i] <= donchian_mid_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if close_4h[i] >= donchian_mid_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals