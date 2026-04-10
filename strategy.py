#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and 1w chop regime filter
# - Primary: 4h price breaking above/below Donchian(20) channel from prior 1d
# - Volume filter: 1d volume > 1.5x 20-period volume MA to confirm participation
# - Regime filter: 1w Choppiness Index(14) > 61.8 (ranging market) to avoid whipsaw in strong trends
# - Exit: Price returns to Donchian midpoint (mean reversion in ranging markets)
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Works in bull/bear: Donchian adapts to volatility, chop filter ensures mean-reversion setup
# - Target: 75-200 total trades over 4 years = 19-50/year for 4h timeframe

name = "4h_1d_1w_donchian_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute HTF data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Donchian(20) from prior 1d (upper/lower channel)
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Align Donchian levels to 4h timeframe (using prior completed 1d bar)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid)
    
    # Calculate 1d volume confirmation: volume > 1.5x 20-period volume MA
    volume_ma_20_1d = pd.Series(volume_1d).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    # Calculate 14-period Choppiness Index for regime filter (using 1w data)
    high_low_1w = high_1w - low_1w
    high_close_1w = np.abs(high_1w - np.roll(close_1w, 1))
    low_close_1w = np.abs(low_1w - np.roll(close_1w, 1))
    
    # Handle first element
    high_low_1w[0] = high_1w[0] - low_1w[0]
    high_close_1w[0] = np.abs(high_1w[0] - close_1w[0])
    low_close_1w[0] = np.abs(low_1w[0] - close_1w[0])
    
    tr_1w = np.maximum(high_low_1w, np.maximum(high_close_1w, low_close_1w))
    atr_sum_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).sum().values
    max_high_1w = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    min_low_1w = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range_hl_1w = max_high_1w - min_low_1w
    range_hl_1w = np.where(range_hl_1w == 0, 1e-10, range_hl_1w)
    
    chop_1w = 100 * np.log10(atr_sum_1w / range_hl_1w) / np.log10(14)
    chop_filter = chop_1w > 61.8  # Chop > 61.8 indicates ranging market
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(donchian_mid_aligned[i]) or np.isnan(volume_ma_20_1d_aligned[i]) or
            np.isnan(chop_1w[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 1d volume > 1.5x 20-period MA
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)
        vol_confirm = vol_1d_current[i] > 1.5 * volume_ma_20_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries at Donchian breakouts
            # Long entry: price breaks above upper channel + vol confirmation + chop filter
            if close[i] > donchian_high_aligned[i] and vol_confirm and chop_filter[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below lower channel + vol confirmation + chop filter
            elif close[i] < donchian_low_aligned[i] and vol_confirm and chop_filter[i]:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit at midpoint (mean reversion)
            # Exit: price returns to Donchian midpoint (mean reversion in ranging markets)
            if position == 1:  # Long position
                if close[i] <= donchian_mid_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if close[i] >= donchian_mid_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals