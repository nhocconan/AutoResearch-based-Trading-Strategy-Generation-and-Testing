#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h volume spike + chop regime filter
# - Primary: 4h price breaks above/below Donchian channel (20-period high/low)
# - HTF: 12h volume > 1.8x 20-period MA for confirmation (avoids low-volume breakouts)
# - Regime filter: 4h Choppiness Index (14) < 38.2 = trending market (trend follow)
# - Long: Price breaks above Donchian upper + volume confirmation + chop trending
# - Short: Price breaks below Donchian lower + volume confirmation + chop trending
# - Exit: Price returns to Donchian midpoint (mean reversion within channel)
# - Position sizing: 0.25 (discrete level, balances return/drawdown, reduces fee churn)
# - Works in bull/bear: Donchian adapts to volatility, volume filters false signals, chop regime targets trending markets
# - Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe

name = "4h_12h_donchian_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 25:  # Need enough data for calculations
        return np.zeros(n)
    
    # Pre-compute 4h data
    close_4h = prices['close'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    volume_4h = prices['volume'].values
    
    # Pre-compute 12h data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 4h Donchian channel (20-period)
    donchian_upper = np.full(len(close_4h), np.nan)
    donchian_lower = np.full(len(close_4h), np.nan)
    donchian_mid = np.full(len(close_4h), np.nan)
    
    for i in range(19, len(close_4h)):
        if not (np.isnan(high_4h[i-19:i+1]).any() or np.isnan(low_4h[i-19:i+1]).any()):
            donchian_upper[i] = np.max(high_4h[i-19:i+1])
            donchian_lower[i] = np.min(low_4h[i-19:i+1])
            donchian_mid[i] = (donchian_upper[i] + donchian_lower[i]) / 2
    
    # Calculate 4h Choppiness Index (14)
    chop = np.full(len(close_4h), np.nan)
    
    # True Range
    tr = np.full(len(close_4h), np.nan)
    for i in range(1, len(close_4h)):
        if not (np.isnan(high_4h[i]) or np.isnan(low_4h[i]) or np.isnan(close_4h[i-1])):
            tr[i] = max(
                high_4h[i] - low_4h[i],
                abs(high_4h[i] - close_4h[i-1]),
                abs(low_4h[i] - close_4h[i-1])
            )
    
    # ATR sum for Chop denominator
    atr_sum = np.full(len(tr), np.nan)
    for i in range(13, len(tr)):
        if not np.isnan(tr[i-13:i+1]).any():
            atr_sum[i] = np.sum(tr[i-13:i+1])
    
    # Choppiness Index
    for i in range(13, len(close_4h)):
        if not (np.isnan(high_4h[i]) or np.isnan(low_4h[i]) or np.isnan(atr_sum[i])):
            highest_high = np.max(high_4h[i-13:i+1])
            lowest_low = np.min(low_4h[i-13:i+1])
            if atr_sum[i] > 0 and (highest_high - lowest_low) > 0:
                chop[i] = 100 * np.log10(atr_sum[i] / (highest_high - lowest_low)) / np.log10(14)
    
    # Calculate 12h volume moving average (20-period)
    volume_ma_20_12h = np.full(len(volume_12h), np.nan)
    for i in range(19, len(volume_12h)):
        if not np.isnan(volume_12h[i-19:i+1]).any():
            volume_ma_20_12h[i] = np.mean(volume_12h[i-19:i+1])
    
    # Align all HTF/LTF indicators to 4h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, prices, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, prices, donchian_lower)
    donchian_mid_aligned = align_htf_to_ltf(prices, prices, donchian_mid)
    chop_aligned = align_htf_to_ltf(prices, prices, chop)
    volume_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_20_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(volume_ma_20_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 12h volume > 1.8x 20-period MA
        volume_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_12h)
        volume_confirm = volume_12h_aligned[i] > 1.8 * volume_ma_20_12h_aligned[i]
        
        # Chop regime filter: CHOP < 38.2 = trending market (good for trend following)
        chop_trending = chop_aligned[i] < 38.2
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above Donchian upper + volume confirmation + chop trending
            if close_4h[i] > donchian_upper_aligned[i] and volume_confirm and chop_trending:
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below Donchian lower + volume confirmation + chop trending
            elif close_4h[i] < donchian_lower_aligned[i] and volume_confirm and chop_trending:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Price returns to Donchian midpoint (mean reversion within channel)
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