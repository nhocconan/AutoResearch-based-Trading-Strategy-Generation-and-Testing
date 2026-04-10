#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and 1w chop regime filter
# - Primary: 4h price breaking above/below 20-period Donchian channels from 1d HTF
# - Volume filter: 1d volume > 1.8x 20-period volume MA to ensure institutional participation
# - Regime filter: 1w Choppiness Index < 38.2 to ensure trending market (avoid chop/range)
# - Exit: Price reverses back to opposite Donchian channel (midpoint for re-entry prevention)
# - Position sizing: 0.25 (discrete level to minimize fee churn while maintaining edge)
# - Target: 100-180 total trades over 4 years = 25-45/year for 4h timeframe
# - Works in bull/bear: Donchian adapts to volatility, volume confirms breakout strength, chop filter avoids false signals in ranging markets

name = "4h_1d_1w_donchian_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 10:
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
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1d Donchian channels (20-period)
    donchian_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_mid_20 = (donchian_high_20 + donchian_low_20) / 2.0
    
    # Align Donchian levels to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_20)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid_20)
    
    # Calculate 1d volume confirmation: volume > 1.8x 20-period volume MA
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    # Calculate 1w Choppiness Index for regime filter
    # Chop = 100 * log10(sum(ATR(14)) / log10(highest_high - lowest_low)) / log10(14)
    high_low_1w = high_1w - low_1w
    high_close_1w = np.abs(high_1w - np.roll(close_1w, 1))
    low_close_1w = np.abs(low_1w - np.roll(close_1w, 1))
    
    high_low_1w[0] = high_1w[0] - low_1w[0]
    high_close_1w[0] = np.abs(high_1w[0] - close_1w[0])
    low_close_1w[0] = np.abs(low_1w[0] - close_1w[0])
    
    tr_1w = np.maximum(high_low_1w, np.maximum(high_close_1w, low_close_1w))
    atr_14_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    
    sum_atr_14 = pd.Series(atr_14_1w).rolling(window=14, min_periods=14).sum().values
    hh_14_1w = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    ll_14_1w = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    range_14_1w = hh_14_1w - ll_14_1w
    
    # Avoid division by zero
    chop_denominator = np.log10(range_14_1w) * np.log10(14)
    chop_denominator = np.where(chop_denominator == 0, 1e-10, chop_denominator)
    chop_1w = 100 * np.log10(sum_atr_14) / chop_denominator
    chop_1w = np.where(range_14_1w == 0, 100, chop_1w)  # Set to 100 when no range
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(donchian_mid_aligned[i]) or np.isnan(volume_ma_20_1d_aligned[i]) or
            np.isnan(chop_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current 1d volume > 1.8x 20-period volume MA
        volume_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)
        vol_confirm = volume_1d_current[i] > 1.8 * volume_ma_20_1d_aligned[i]
        
        # Regime filter: Chop < 38.2 to ensure trending market
        trending_market = chop_1w_aligned[i] < 38.2
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Donchian HIGH + vol confirmation + trending market
            if (close[i] > donchian_high_aligned[i] and 
                vol_confirm and trending_market):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Donchian LOW + vol confirmation + trending market
            elif (close[i] < donchian_low_aligned[i] and 
                  vol_confirm and trending_market):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: price returns to opposite Donchian level (midpoint prevents whipsaw)
            if position == 1:  # Long position
                if close[i] < donchian_mid_aligned[i]:  # Exit when price crosses below midpoint
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if close[i] > donchian_mid_aligned[i]:  # Exit when price crosses above midpoint
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals