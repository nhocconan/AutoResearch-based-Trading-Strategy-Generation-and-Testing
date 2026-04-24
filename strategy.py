#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d volume confirmation and choppiness regime filter.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d for volume confirmation (above 20-period average) and choppiness regime (CHOP > 61.8 = ranging).
- Entry: Long when price breaks above Donchian(20) high AND 1d volume > 1.2 * 20-period average volume AND 1d CHOP > 61.8.
         Short when price breaks below Donchian(20) low AND 1d volume > 1.2 * 20-period average volume AND 1d CHOP > 61.8.
- Exit: Opposite Donchian breakout (price crosses opposite Donchian level) OR choppiness regime shifts to trending (CHOP < 38.2).
- Signal size: 0.25 discrete to minimize fee drag while maintaining profit potential.
- Donchian channels provide objective breakout levels in ranging markets.
- Volume confirmation ensures breakouts have participation.
- Choppiness regime filter avoids false breakouts in strong trends and captures mean-reversion in ranges.
- Works in bull markets (buy breakouts in ranges) and bear markets (sell breakdowns in ranges).
- Estimated trades: ~100 total over 4 years (~25/year) based on Donchian breakout frequency with filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 12h Donchian channels (20-period)
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback-1, n):
        highest_high[i] = np.max(high[i-lookback+1:i+1])
        lowest_low[i] = np.min(low[i-lookback+1:i+1])
    
    # Calculate 1d HTF data for volume confirmation and choppiness
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d volume confirmation: volume > 1.2 * 20-period average volume
    volume_20_avg = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    volume_confirm = df_1d['volume'].values > (1.2 * volume_20_avg)
    volume_confirm_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm, additional_delay_bars=0)
    
    # 1d choppiness index: CHOP = 100 * log10(sum(ATR(14)) / log10(highest_high - lowest_low) / sqrt(14))
    # Simplified: CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    # We'll use a proxy: price position within Donchian(20) on 1d
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    
    # Calculate 1d Donchian(20)
    highest_high_1d = np.full(len(df_1d), np.nan)
    lowest_low_1d = np.full(len(df_1d), np.nan)
    for i in range(lookback-1, len(df_1d)):
        highest_high_1d[i] = np.max(df_1d_high[i-lookback+1:i+1])
        lowest_low_1d[i] = np.min(df_1d_low[i-lookback+1:i+1])
    
    # Calculate choppiness proxy: % position within 1d Donchian channel
    # 0% = at low, 100% = at high, 50% = middle
    chop_proxy = np.full(len(df_1d), np.nan)
    for i in range(lookback-1, len(df_1d)):
        if highest_high_1d[i] != lowest_low_1d[i]:
            chop_proxy[i] = (df_1d_close[i] - lowest_low_1d[i]) / (highest_high_1d[i] - lowest_low_1d[i]) * 100
        else:
            chop_proxy[i] = 50
    
    # Choppiness regime: ranging when price is in middle 40-60% of channel
    chop_ranging = (chop_proxy >= 40) & (chop_proxy <= 60)
    chop_ranging_aligned = align_htf_to_ltf(prices, df_1d, chop_ranging, additional_delay_bars=0)
    
    # Trending regime: when price is in outer 20%-80% (we'll use opposite for exit)
    chop_trending = (chop_proxy < 30) | (chop_proxy > 70)
    chop_trending_aligned = align_htf_to_ltf(prices, df_1d, chop_trending, additional_delay_bars=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = lookback - 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(volume_confirm_aligned[i]) or
            np.isnan(chop_ranging_aligned[i]) or np.isnan(chop_trending_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        # Exit conditions: opposite Donchian breakout OR choppiness regime shifts to trending
        if position != 0:
            # Exit long: price breaks below Donchian low OR choppiness shifts to trending
            if position == 1:
                if curr_close < lowest_low[i] or chop_trending_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: price breaks above Donchian high OR choppiness shifts to trending
            elif position == -1:
                if curr_close > highest_high[i] or chop_trending_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout with volume confirmation AND choppy ranging regime
        if position == 0:
            # Long: price breaks above Donchian high AND volume confirmation AND choppy ranging
            if curr_close > highest_high[i] and volume_confirm_aligned[i] and chop_ranging_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND volume confirmation AND choppy ranging
            elif curr_close < lowest_low[i] and volume_confirm_aligned[i] and chop_ranging_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dVolume_ChopRegime_v1"
timeframe = "12h"
leverage = 1.0