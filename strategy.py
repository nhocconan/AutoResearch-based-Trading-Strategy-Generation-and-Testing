#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Choppiness Index regime filter + 12h Donchian breakout + 1d volume confirmation
# - Choppiness Index (14) determines market regime: >61.8 = range (mean-revert), <38.2 = trend (follow breakout)
# - In trending regime (CHOP < 38.2): long on Donchian(20) breakout above, short on breakdown below
# - In ranging regime (CHOP > 61.8): long at Donchian(20) lower band, short at upper band (mean reversion)
# - Volume confirmation: current 12h volume > 1.5x 1d average volume (scaled to 12h) for conviction
# - Designed to work in both bull and bear markets by adapting to regime
# - Target: 15-30 trades/year to minimize fee drag on 12h timeframe

name = "12h_Chop_Donchian_1dVolume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume confirmation (using daily volume as reference)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d volume average (20-period) - represents normal daily volume
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Choppiness Index (14) on 12h data
    # CHOP = 100 * log10(sum(TR(14)) / (max(HH,14) - min(LL,14))) / log10(14)
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    
    # Avoid division by zero
    range_hl = highest_high - lowest_low
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)
    
    chop = 100 * np.log10(atr_sum / range_hl) / np.log10(14)
    chop = chop.fillna(50).values  # neutral when undefined
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for all indicators (max of 20,14)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(chop[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_ma_1d_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 12h volume > 1.5x 1d average volume (scaled)
        # Scale 1d average to 12h: 1d = 2x 12h bars, so multiply by 2
        volume_filter = vol_ma_1d_aligned[i] > 0 and volume[i] > 1.5 * (vol_ma_1d_aligned[i] * 2.0)
        
        if position == 0:
            # Determine regime based on Choppiness Index
            if chop[i] < 38.2:  # Trending regime
                # Long on breakout above Donchian high
                if close[i] > donchian_high[i] and volume_filter:
                    signals[i] = 0.25
                    position = 1
                # Short on breakdown below Donchian low
                elif close[i] < donchian_low[i] and volume_filter:
                    signals[i] = -0.25
                    position = -1
            elif chop[i] > 61.8:  # Ranging regime
                # Mean reversion: long at support (Donchian low)
                if close[i] <= donchian_low[i] and volume_filter:
                    signals[i] = 0.25
                    position = 1
                # Short at resistance (Donchian high)
                elif close[i] >= donchian_high[i] and volume_filter:
                    signals[i] = -0.25
                    position = -1
            # In between 38.2-61.8: neutral/choppy, no new entries
            
        elif position == 1:
            # Long position management
            if chop[i] < 38.2:  # Trending: exit on breakdown
                if close[i] < donchian_low[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif chop[i] > 61.8:  # Ranging: exit at resistance or opposite signal
                if close[i] >= donchian_high[i] or (chop[i] < 38.2 and close[i] > donchian_low[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # Neutral zone: exit on opposite Donchian touch
                if close[i] <= donchian_low[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
                    
        elif position == -1:
            # Short position management
            if chop[i] < 38.2:  # Trending: exit on breakout
                if close[i] > donchian_high[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            elif chop[i] > 61.8:  # Ranging: exit at support or opposite signal
                if close[i] <= donchian_low[i] or (chop[i] > 61.8 and close[i] < donchian_high[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:  # Neutral zone: exit on opposite Donchian touch
                if close[i] >= donchian_high[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals