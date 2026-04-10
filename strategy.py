#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w volume confirmation and 1d chop regime filter
# - Primary: 12h price breaking above/below Donchian(20) channel from prior 1w
# - Volume filter: 1w volume > 1.5x 20-period volume MA to confirm institutional participation
# - Regime filter: 1d Choppiness Index(14) > 61.8 (ranging market) to avoid whipsaw in strong trends
# - Exit: Price returns to Donchian midpoint or opposite channel touch
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Works in bull/bear: Donchian adapts to volatility, chop filter ensures mean-reversion setup
# - Target: 50-150 total trades over 4 years = 12-37/year for 12h timeframe

name = "12h_1w_1d_donchian_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute HTF data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian(20) from prior 1w (upper/lower channel)
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Align Donchian levels to 12h timeframe (using prior completed 1w bar)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1w, donchian_mid)
    
    # Calculate 1w volume confirmation: volume > 1.5x 20-period volume MA
    volume_ma_20_1w = pd.Series(volume_1w).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_ma_20_1w)
    
    # Calculate 14-period Choppiness Index for regime filter (using 1d data)
    high_low_1d = high_1d - low_1d
    high_close_1d = np.abs(high_1d - np.roll(close_1d, 1))
    low_close_1d = np.abs(low_1d - np.roll(close_1d, 1))
    
    # Handle first element
    high_low_1d[0] = high_1d[0] - low_1d[0]
    high_close_1d[0] = np.abs(high_1d[0] - close_1d[0])
    low_close_1d[0] = np.abs(low_1d[0] - close_1d[0])
    
    tr_1d = np.maximum(high_low_1d, np.maximum(high_close_1d, low_close_1d))
    atr_sum_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    max_high_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range_hl_1d = max_high_1d - min_low_1d
    range_hl_1d = np.where(range_hl_1d == 0, 1e-10, range_hl_1d)
    
    chop_1d = 100 * np.log10(atr_sum_1d / range_hl_1d) / np.log10(14)
    chop_filter = chop_1d > 61.8  # Chop > 61.8 indicates ranging market
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(donchian_mid_aligned[i]) or np.isnan(volume_ma_20_1w_aligned[i]) or
            np.isnan(chop_1d[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 1w volume > 1.5x 20-period MA
        vol_1w_current = align_htf_to_ltf(prices, df_1w, volume_1w)
        vol_confirm = vol_1w_current[i] > 1.5 * volume_ma_20_1w_aligned[i]
        
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
        else:  # Have position - look for exit at midpoint or opposite channel
            # Exit: price reaches midpoint (mean reversion) or touches opposite channel (reversal)
            if position == 1:  # Long position
                if close[i] <= donchian_mid_aligned[i] or close[i] >= donchian_high_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if close[i] >= donchian_mid_aligned[i] or close[i] <= donchian_low_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals