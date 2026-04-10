#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d volume confirmation and chop regime filter
# - Primary: 4h price breaks above/below Donchian channel (20-period) from prior 4h session
# - HTF: 1d volume > 1.3x 20-period MA for confirmation (avoids low-volume breakouts)
# - Regime filter: 4h Choppiness Index (14) < 38.2 to ensure trending market (avoids chop)
# - Long: Close > Upper Donchian + volume confirmation + chop trending
# - Short: Close < Lower Donchian + volume confirmation + chop trending
# - Exit: Close crosses back inside Donchian channel OR chop regime shifts to ranging (CHOP > 61.8)
# - Position sizing: 0.25 (discrete level, balances return/drawdown, reduces fee churn)
# - Works in bull/bear: Donchian adapts to volatility, volume filters false breakouts, chop regime avoids whipsaws
# - Target: 80-150 total trades over 4 years (20-38/year) for 4h timeframe

name = "4h_1d_donchian_volume_chop_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 25:  # Need enough data for calculations
        return np.zeros(n)
    
    # Pre-compute 4h data
    close_4h = prices['close'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    
    # Pre-compute 1d data
    volume_1d = df_1d['volume'].values
    
    # Calculate 4h Donchian channel (20-period) - using prior period only
    donchian_high = np.full(len(close_4h), np.nan)
    donchian_low = np.full(len(close_4h), np.nan)
    
    for i in range(20, len(close_4h)):
        if not (np.isnan(high_4h[i-20:i]).any() or np.isnan(low_4h[i-20:i]).any()):
            donchian_high[i] = np.max(high_4h[i-20:i])  # Prior 20 periods
            donchian_low[i] = np.min(low_4h[i-20:i])   # Prior 20 periods
    
    # Calculate 4h Choppiness Index (14)
    chop = np.full(len(close_4h), np.nan)
    atr_sum = np.full(len(close_4h), np.nan)
    
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
    
    # Calculate 1d volume moving average (20-period)
    volume_ma_20_1d = np.full(len(volume_1d), np.nan)
    for i in range(19, len(volume_1d)):
        if not np.isnan(volume_1d[i-19:i+1]).any():
            volume_ma_20_1d[i] = np.mean(volume_1d[i-19:i+1])
    
    # Align all HTF/LTF indicators to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, prices, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, prices, donchian_low)
    chop_aligned = align_htf_to_ltf(prices, prices, chop)
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(volume_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.3x 20-period MA
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        volume_confirm = volume_1d_aligned[i] > 1.3 * volume_ma_20_1d_aligned[i]
        
        # Chop regime filter: CHOP < 38.2 = trending, CHOP > 61.8 = ranging
        chop_trending = chop_aligned[i] < 38.2
        chop_ranging = chop_aligned[i] > 61.8
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Close > Upper Donchian + volume confirmation + chop trending
            if close_4h[i] > donchian_high_aligned[i] and volume_confirm and chop_trending:
                position = 1
                signals[i] = 0.25
            # Short entry: Close < Lower Donchian + volume confirmation + chop trending
            elif close_4h[i] < donchian_low_aligned[i] and volume_confirm and chop_trending:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Close crosses back inside Donchian channel OR chop regime shifts to ranging
            if position == 1:  # Long position
                if close_4h[i] < donchian_high_aligned[i] or chop_ranging:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if close_4h[i] > donchian_low_aligned[i] or chop_ranging:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals