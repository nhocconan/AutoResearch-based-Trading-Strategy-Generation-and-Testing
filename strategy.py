#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d volume confirmation and chop regime filter
# - Primary: 12h timeframe to capture medium-term trends while minimizing fees
# - HTF: 1d for Williams Alligator (jaw/teeth/lips) and volume confirmation
# - Williams Alligator: Jaw (SMA13 shifted 8), Teeth (SMA8 shifted 5), Lips (SMA5 shifted 3)
# - Long: Lips > Teeth > Jaw (bullish alignment) + 1d volume > 1.5x 20-period MA + chop < 50 (trending)
# - Short: Lips < Teeth < Jaw (bearish alignment) + 1d volume > 1.5x 20-period MA + chop < 50 (trending)
# - Exit: When Alligator lines cross (Lips/Teeth or Teeth/Jaw) indicating trend weakness
# - Position sizing: 0.25 (discrete level)
# - Target: 50-150 total trades over 4 years (12-37/year) - within 12h sweet spot
# - Works in bull/bear: Alligator catches strong trends; chop filter avoids whipsaws in ranging markets (2025)

name = "12h_1d_alligator_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 12h OHLCV
    open_12h = prices['open'].values
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    volume_12h = prices['volume'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Williams Alligator on 1d
    # Jaw: SMA13 shifted 8 bars
    jaw = pd.Series(close_1d).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth: SMA8 shifted 5 bars
    teeth = pd.Series(close_1d).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips: SMA5 shifted 3 bars
    lips = pd.Series(close_1d).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align Alligator lines to 12h
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Calculate 1d Choppiness Index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / (log10(highest_high - lowest_low) * n))
    # Simplified: CHOP = 100 * log10(atr_sum / (log10(range) * period)) / log10(period)
    tr1 = pd.Series(high_1d).shift(1) - pd.Series(low_1d).shift(1)
    tr2 = abs(pd.Series(high_1d) - pd.Series(close_1d).shift(1))
    tr3 = abs(pd.Series(low_1d) - pd.Series(close_1d).shift(1))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr_1d.rolling(window=14, min_periods=14).mean().values
    
    # Calculate sum of ATR over 14 periods
    atr_sum = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    # Calculate highest high and lowest low over 14 periods
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    price_range = highest_high - lowest_low
    price_range = np.where(price_range == 0, 1e-10, price_range)
    
    # Choppiness Index formula
    chop_raw = 100 * np.log10(atr_sum / (np.log10(price_range) * 14)) / np.log10(14)
    chop_raw = np.where(np.isnan(chop_raw), 50, chop_raw)  # Default to 50 if invalid
    chop_raw = np.clip(chop_raw, 0, 100)  # Clamp to valid range
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_raw)
    
    # Calculate 1d volume moving average (20-period) for volume confirmation
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Warmup period: need enough data for Alligator calculations
    # Lips needs 5 + 3 = 8 bars, Teeth needs 8 + 5 = 13, Jaw needs 13 + 8 = 21
    warmup = 25
    
    for i in range(warmup, n):
        # Skip if any required data is invalid
        if (np.isnan(lips_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(jaw_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(volume_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime conditions
        # Trending regime: CHOP < 50 (below 50 indicates trending market)
        trending_regime = chop_aligned[i] < 50
        
        # Volume confirmation: current 1d volume > 1.5x 20-period MA
        volume_spike = volume_1d[i] > 1.5 * volume_ma_20_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Bullish alignment: Lips > Teeth > Jaw
            bullish = (lips_aligned[i] > teeth_aligned[i] and 
                      teeth_aligned[i] > jaw_aligned[i])
            # Bearish alignment: Lips < Teeth < Jaw
            bearish = (lips_aligned[i] < teeth_aligned[i] and 
                      teeth_aligned[i] < jaw_aligned[i])
            
            # Long entry: Bullish alignment + trending regime + volume spike
            if bullish and trending_regime and volume_spike:
                position = 1
                signals[i] = 0.25
            # Short entry: Bearish alignment + trending regime + volume spike
            elif bearish and trending_regime and volume_spike:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: When Alligator lines cross (trend weakening)
            if position == 1:  # Long position
                # Exit when Lips crosses below Teeth OR Teeth crosses below Jaw
                exit_condition = (
                    lips_aligned[i] < teeth_aligned[i] or 
                    teeth_aligned[i] < jaw_aligned[i]
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                # Exit when Lips crosses above Teeth OR Teeth crosses above Jaw
                exit_condition = (
                    lips_aligned[i] > teeth_aligned[i] or 
                    teeth_aligned[i] > jaw_aligned[i]
                )
                if exit_condition:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals