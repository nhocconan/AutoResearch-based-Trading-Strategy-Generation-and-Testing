#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Choppiness Index regime filter + 12-hour Donchian breakout with volume confirmation.
# In choppy markets (CHOP > 61.8): mean reversion at Donchian bands (sell upper, buy lower).
# In trending markets (CHOP < 38.2): trend following (buy upper breakout, sell lower breakdown).
# Uses 12h Donchian for structure, 12h Choppiness for regime, volume spike for confirmation.
# Designed to work in both bull (trend following) and bear (mean reversion in chop) markets.
name = "4h_12hDonchian_ChopRegime_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12-hour data for Donchian bands and Choppiness Index
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    vol_12h = df_12h['volume'].values
    
    # Calculate 12-period Donchian channels
    donch_high = pd.Series(high_12h).rolling(window=12, min_periods=12).max().values
    donch_low = pd.Series(low_12h).rolling(window=12, min_periods=12).min().values
    
    # Calculate 14-period Choppiness Index
    # TR = max(high-low, abs(high-previous close), abs(low-previous close))
    tr1 = np.abs(high_12h - low_12h)
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high_12h[0] - low_12h[0]  # first TR
    
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(TR14)/(max(high14)-min(low14))) / log10(14)
    sum_tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    range14 = highest_high - lowest_low
    chop = 100 * np.log10(sum_tr14 / range14) / np.log10(14)
    chop = np.where(range14 == 0, 50, chop)  # avoid division by zero
    
    # 20-period volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all 12h indicators to 4h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_12h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_12h, donch_low)
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop)
    vol_12h_ma = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_12h_ma_aligned = align_htf_to_ltf(prices, df_12h, vol_12h_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(vol_ma_20[i]) or
            np.isnan(vol_12h_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        donch_high_val = donch_high_aligned[i]
        donch_low_val = donch_low_aligned[i]
        chop_val = chop_aligned[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        vol_12h_ma_val = vol_12h_ma_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = vol > 1.5 * vol_ma
        
        if position == 0:
            # Determine market regime based on Choppiness Index
            if chop_val > 61.8:  # Choppy market - mean reversion
                # Buy near lower Donchian band, sell near upper band
                if price <= donch_low_val and vol_confirm:
                    signals[i] = 0.25
                    position = 1
                elif price >= donch_high_val and vol_confirm:
                    signals[i] = -0.25
                    position = -1
            else:  # Trending market (chop < 61.8) - trend following
                # Buy breakout above upper band, sell breakdown below lower band
                if price > donch_high_val and vol_confirm:
                    signals[i] = 0.25
                    position = 1
                elif price < donch_low_val and vol_confirm:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit conditions
            if chop_val > 61.8:  # In chop: take profit at upper band
                if price >= donch_high_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # In trend: trail with lower Donchian band
                if price < donch_low_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        
        elif position == -1:
            # Short exit conditions
            if chop_val > 61.8:  # In chop: take profit at lower band
                if price <= donch_low_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:  # In trend: trail with upper Donchian band
                if price > donch_high_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals