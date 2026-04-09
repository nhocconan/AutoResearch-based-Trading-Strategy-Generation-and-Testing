#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Donchian breakout with volume confirmation and choppiness regime filter
# - Uses 1d HTF for Donchian channel (20-period) to identify structural breakouts
# - Volume confirmation: current 4h volume > 1.5x 20-period average to avoid low-volume false signals
# - Choppiness regime filter: 1d Choppiness Index (14) < 38.2 = trending market (favor breakouts)
# - Fixed position size 0.25 to control drawdown (BTC 2022 drawdown: 0.25*77% = ~19% loss)
# - Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years)

name = "4h_1d_donchian_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian channel (20 periods)
    donch_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d Choppiness Index (14 periods)
    # CHOP = 100 * log10(sum(ATR(1)) / (n * (HHV - LLV))) / log10(n)
    tr1 = np.maximum(high_1d - low_1d, 
                     np.absolute(high_1d - np.roll(close_1d, 1)),
                     np.absolute(low_1d - np.roll(close_1d, 1)))
    tr1[0] = high_1d[0] - low_1d[0]  # first period
    atr1 = pd.Series(tr1).rolling(window=1, min_periods=1).sum().values
    sum_atr1 = pd.Series(atr1).rolling(window=14, min_periods=14).sum().values
    hhvl = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values - \
           pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr1 / (hhvl + 1e-10)) / np.log10(14)
    
    # Align all HTF data to 4h timeframe (wait for completed HTF bar)
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high_20)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low_20)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Pre-compute volume confirmation (20-period average for 4h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(vol_ma_20[i]) or
            vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # Choppiness regime filter: CHOP < 38.2 = trending market
        trending_market = chop_aligned[i] < 38.2
        
        # Fixed position size
        position_size = 0.25
        
        if position == 1:  # Long position
            # Exit conditions: price breaks below Donchian low or trend changes to ranging
            if close[i] < donch_low_aligned[i] or not trending_market:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
                
        elif position == -1:  # Short position
            # Exit conditions: price breaks above Donchian high or trend changes to ranging
            if close[i] > donch_high_aligned[i] or not trending_market:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
        else:  # Flat
            # Entry logic: Donchian breakout with volume confirmation in trending market
            if volume_confirmed and trending_market:
                if close[i] > donch_high_aligned[i]:
                    # Bullish breakout
                    position = 1
                    signals[i] = position_size
                elif close[i] < donch_low_aligned[i]:
                    # Bearish breakout
                    position = -1
                    signals[i] = -position_size
    
    return signals