#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d volume spike + chop regime filter
# - Primary: 12h Donchian breakout (20-period) for trend continuation
# - HTF: 1d volume > 2.0x 24-period MA for institutional participation confirmation
# - Regime filter: 12h Choppiness Index (14) < 38.2 = trending market (breakout continuation)
# - Long: Price breaks above Donchian(20) upper + volume confirmation + chop trending
# - Short: Price breaks below Donchian(20) lower + volume confirmation + chop trending
# - Exit: Price crosses Donchian(20) midpoint (mean reversion to median)
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Works in bull/bear: Donchian captures breakouts in trending markets, volume filters weak moves, chop filter avoids false breakouts in ranging markets
# - Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe

name = "12h_1d_donchian_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
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
    
    # Calculate 12h Donchian channels (20-period)
    donchian_upper = np.full(len(close), np.nan)
    donchian_lower = np.full(len(close), np.nan)
    donchian_mid = np.full(len(close), np.nan)
    
    for i in range(19, len(close)):
        if not (np.isnan(high[i-19:i+1]).any() or np.isnan(low[i-19:i+1]).any()):
            donchian_upper[i] = np.max(high[i-19:i+1])
            donchian_lower[i] = np.min(low[i-19:i+1])
            donchian_mid[i] = (donchian_upper[i] + donchian_lower[i]) / 2.0
    
    # Calculate 12h Choppiness Index (14)
    chop = np.full(len(close), np.nan)
    
    # True Range
    tr = np.full(len(close), np.nan)
    for i in range(1, len(close)):
        if not (np.isnan(high[i]) or np.isnan(low[i]) or np.isnan(close[i-1])):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
    
    # ATR sum for Chop denominator
    atr_sum = np.full(len(tr), np.nan)
    for i in range(13, len(tr)):
        if not np.isnan(tr[i-13:i+1]).any():
            atr_sum[i] = np.sum(tr[i-13:i+1])
    
    # Choppiness Index
    for i in range(13, len(close)):
        if not (np.isnan(high[i]) or np.isnan(low[i]) or np.isnan(atr_sum[i])):
            highest_high = np.max(high[i-13:i+1])
            lowest_low = np.min(low[i-13:i+1])
            if atr_sum[i] > 0 and (highest_high - lowest_low) > 0:
                chop[i] = 100 * np.log10(atr_sum[i] / (highest_high - lowest_low)) / np.log10(14)
    
    # Calculate 1d volume moving average (24-period)
    volume_ma_24_1d = np.full(len(volume_1d), np.nan)
    for i in range(23, len(volume_1d)):
        if not np.isnan(volume_1d[i-23:i+1]).any():
            volume_ma_24_1d[i] = np.mean(volume_1d[i-23:i+1])
    
    # Align HTF indicators to 12h timeframe
    volume_ma_24_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_24_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(chop[i]) or 
            np.isnan(volume_ma_24_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 2.0x 24-period MA
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        volume_confirm = volume_1d_aligned[i] > 2.0 * volume_ma_24_1d_aligned[i]
        
        # Chop regime filter: CHOP < 38.2 = trending market (breakout continuation)
        chop_trending = chop[i] < 38.2
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above Donchian upper + volume confirmation + chop trending
            if close[i] > donchian_upper[i] and volume_confirm and chop_trending:
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below Donchian lower + volume confirmation + chop trending
            elif close[i] < donchian_lower[i] and volume_confirm and chop_trending:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Price crosses Donchian midpoint (mean reversion to median)
            if position == 1:  # Long position
                if close[i] <= donchian_mid[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if close[i] >= donchian_mid[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals