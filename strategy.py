#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w volume spike and chop regime filter
# - Primary: 1d price breaks above/below Donchian channel (20-period high/low)
# - HTF: 1w volume > 2.0x 20-period MA for confirmation (avoids low-volume breakouts)
# - Regime filter: 1d Choppiness Index (14) < 38.2 = trending market (trend follow)
# - Long: Price breaks above Donchian upper + volume confirmation + chop trending
# - Short: Price breaks below Donchian lower + volume confirmation + chop trending
# - Exit: Price returns to Donchian midpoint (mean reversion within channel)
# - Position sizing: 0.25 (discrete level, balances return/drawdown, reduces fee churn)
# - Works in bull/bear: Donchian adapts to volatility, volume filters false signals, chop regime targets trending markets
# - Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe

name = "1d_1w_donchian_volume_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 25:  # Need enough data for calculations
        return np.zeros(n)
    
    # Pre-compute 1d data
    close_1d = prices['close'].values
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    volume_1d = prices['volume'].values
    
    # Pre-compute 1w data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate 1d Donchian channel (20-period)
    donchian_upper = np.full(len(close_1d), np.nan)
    donchian_lower = np.full(len(close_1d), np.nan)
    donchian_mid = np.full(len(close_1d), np.nan)
    
    for i in range(19, len(close_1d)):
        if not (np.isnan(high_1d[i-19:i+1]).any() or np.isnan(low_1d[i-19:i+1]).any()):
            donchian_upper[i] = np.max(high_1d[i-19:i+1])
            donchian_lower[i] = np.min(low_1d[i-19:i+1])
            donchian_mid[i] = (donchian_upper[i] + donchian_lower[i]) / 2
    
    # Calculate 1d Choppiness Index (14)
    chop = np.full(len(close_1d), np.nan)
    
    # True Range
    tr = np.full(len(close_1d), np.nan)
    for i in range(1, len(close_1d)):
        if not (np.isnan(high_1d[i]) or np.isnan(low_1d[i]) or np.isnan(close_1d[i-1])):
            tr[i] = max(
                high_1d[i] - low_1d[i],
                abs(high_1d[i] - close_1d[i-1]),
                abs(low_1d[i] - close_1d[i-1])
            )
    
    # ATR sum for Chop denominator
    atr_sum = np.full(len(tr), np.nan)
    for i in range(13, len(tr)):
        if not np.isnan(tr[i-13:i+1]).any():
            atr_sum[i] = np.sum(tr[i-13:i+1])
    
    # Choppiness Index
    for i in range(13, len(close_1d)):
        if not (np.isnan(high_1d[i]) or np.isnan(low_1d[i]) or np.isnan(atr_sum[i])):
            highest_high = np.max(high_1d[i-13:i+1])
            lowest_low = np.min(low_1d[i-13:i+1])
            if atr_sum[i] > 0 and (highest_high - lowest_low) > 0:
                chop[i] = 100 * np.log10(atr_sum[i] / (highest_high - lowest_low)) / np.log10(14)
    
    # Calculate 1w volume moving average (20-period)
    volume_ma_20_1w = np.full(len(volume_1w), np.nan)
    for i in range(19, len(volume_1w)):
        if not np.isnan(volume_1w[i-19:i+1]).any():
            volume_ma_20_1w[i] = np.mean(volume_1w[i-19:i+1])
    
    # Align all HTF/LTF indicators to 1d timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, prices, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, prices, donchian_lower)
    donchian_mid_aligned = align_htf_to_ltf(prices, prices, donchian_mid)
    chop_aligned = align_htf_to_ltf(prices, prices, chop)
    volume_ma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_ma_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(volume_ma_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1w volume > 2.0x 20-period MA
        volume_1w_aligned = align_htf_to_ltf(prices, df_1w, volume_1w)
        volume_confirm = volume_1w_aligned[i] > 2.0 * volume_ma_20_1w_aligned[i]
        
        # Chop regime filter: CHOP < 38.2 = trending market (good for trend following)
        chop_trending = chop_aligned[i] < 38.2
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above Donchian upper + volume confirmation + chop trending
            if close_1d[i] > donchian_upper_aligned[i] and volume_confirm and chop_trending:
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below Donchian lower + volume confirmation + chop trending
            elif close_1d[i] < donchian_lower_aligned[i] and volume_confirm and chop_trending:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Price returns to Donchian midpoint (mean reversion within channel)
            if position == 1:  # Long position
                if close_1d[i] <= donchian_mid_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if close_1d[i] >= donchian_mid_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals