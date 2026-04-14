#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Choppiness Index + Donchian Breakout with Volume Filter
# Uses 12h Choppiness Index to detect trend regimes (CHOP < 38.2 = trending, CHOP > 61.8 = ranging)
# In trending markets: enter on Donchian(20) breakout with volume confirmation
# In ranging markets: fade Donchian breakouts (mean reversion at bands)
# Volume filter: require 1d volume > 1.5x 20-period average to confirm breakout strength
# Designed to work in both bull/bear by adapting to market regime
# Target: 25-40 trades per year per symbol (100-160 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h and 1d data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h Choppiness Index (14-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range for CHOP
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR (14-period smoothed TR)
    atr = pd.Series(tr).ewm(span=14, adjust=False).mean().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum(TR14) / (HH - LL)) / log10(14)
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(tr_sum / (hh - ll)) / np.log10(14)
    
    # Calculate 12h Donchian channels (20-period)
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Calculate 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop)
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_12h, donchian_mid)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations (need 34 for Chop: 14+20)
    start = 34
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(chop_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        chop_val = chop_aligned[i]
        vol_1d_current = vol_1d[i] if i < len(vol_1d) else vol_1d[-1]
        
        if position == 0:
            # Determine market regime
            is_trending = chop_val < 38.2
            is_ranging = chop_val > 61.8
            
            if is_trending:
                # Trending market: breakout strategy
                if (price > donchian_high_aligned[i] and 
                    vol_1d_current > 1.5 * vol_ma_1d_aligned[i]):
                    position = 1
                    signals[i] = position_size
                elif (price < donchian_low_aligned[i] and 
                      vol_1d_current > 1.5 * vol_ma_1d_aligned[i]):
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = 0.0
            elif is_ranging:
                # Ranging market: mean reversion at Donchian bands
                if price < donchian_low_aligned[i] and vol_1d_current > vol_ma_1d_aligned[i]:
                    position = 1  # Buy at lower band
                    signals[i] = position_size
                elif price > donchian_high_aligned[i] and vol_1d_current > vol_ma_1d_aligned[i]:
                    position = -1  # Sell at upper band
                    signals[i] = -position_size
                else:
                    signals[i] = 0.0
            else:
                # Transition zone: no trade
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses Donchian mid or volatility drops
            if price > donchian_mid_aligned[i] or vol_1d_current < vol_ma_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses Donchian mid or volatility drops
            if price < donchian_mid_aligned[i] or vol_1d_current < vol_ma_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Chop_Donchian_Breakout_Volume"
timeframe = "12h"
leverage = 1.0