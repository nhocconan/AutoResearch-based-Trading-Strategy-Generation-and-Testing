#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d volume confirmation and chop regime filter
# - Primary: 12h price above/below Alligator Jaw/Teeth/Lips for trend direction
# - HTF: 1d volume > 1.5x 20-period MA for confirmation (avoids low-volume breakouts)
# - Regime filter: 12h Choppiness Index (14) < 38.2 to ensure trending market
# - Long: Price > Alligator Lips + volume confirmation + chop trending
# - Short: Price < Alligator Jaw + volume confirmation + chop trending
# - Exit: Price crosses back inside Alligator mouth OR chop regime shifts to ranging
# - Position sizing: 0.25 (discrete level, balances return/drawdown, reduces fee churn)
# - Works in bull/bear: Alligator adapts to volatility, volume filters false signals, chop regime avoids whipsaws
# - Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe

name = "12h_1d_alligator_volume_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 25:  # Need enough data for calculations
        return np.zeros(n)
    
    # Pre-compute 12h data
    close_12h = prices['close'].values
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Williams Alligator (13,8,5) - using prior close only
    # Jaw (13-period SMMA of median price, shifted 8 bars)
    # Teeth (8-period SMMA of median price, shifted 5 bars)
    # Lips (5-period SMMA of median price, shifted 3 bars)
    median_price_1d = (high_1d + low_1d) / 2
    
    # SMMA calculation (smoothed moving average)
    def smma(source, period):
        result = np.full_like(source, np.nan)
        if len(source) >= period:
            # First value is simple SMA
            result[period-1] = np.mean(source[:period])
            # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT) / period
            for i in range(period, len(source)):
                if not np.isnan(source[i]) and not np.isnan(result[i-1]):
                    result[i] = (result[i-1] * (period-1) + source[i]) / period
        return result
    
    jaw_raw = smma(median_price_1d, 13)
    teeth_raw = smma(median_price_1d, 8)
    lips_raw = smma(median_price_1d, 5)
    
    # Apply shifts: Jaw shifted 8, Teeth shifted 5, Lips shifted 3
    alligator_jaw = np.full_like(jaw_raw, np.nan)
    alligator_teeth = np.full_like(teeth_raw, np.nan)
    alligator_lips = np.full_like(lips_raw, np.nan)
    
    for i in range(len(jaw_raw)):
        if i >= 8 and not np.isnan(jaw_raw[i]):
            alligator_jaw[i-8] = jaw_raw[i]
        if i >= 5 and not np.isnan(teeth_raw[i]):
            alligator_teeth[i-5] = teeth_raw[i]
        if i >= 3 and not np.isnan(lips_raw[i]):
            alligator_lips[i-3] = lips_raw[i]
    
    # Calculate 12h Choppiness Index (14)
    chop = np.full(len(close_12h), np.nan)
    
    # True Range
    tr = np.full(len(close_12h), np.nan)
    for i in range(1, len(close_12h)):
        if not (np.isnan(high_12h[i]) or np.isnan(low_12h[i]) or np.isnan(close_12h[i-1])):
            tr[i] = max(
                high_12h[i] - low_12h[i],
                abs(high_12h[i] - close_12h[i-1]),
                abs(low_12h[i] - close_12h[i-1])
            )
    
    # ATR sum for Chop denominator
    atr_sum = np.full(len(tr), np.nan)
    for i in range(13, len(tr)):
        if not np.isnan(tr[i-13:i+1]).any():
            atr_sum[i] = np.sum(tr[i-13:i+1])
    
    # Choppiness Index
    for i in range(13, len(close_12h)):
        if not (np.isnan(high_12h[i]) or np.isnan(low_12h[i]) or np.isnan(atr_sum[i])):
            highest_high = np.max(high_12h[i-13:i+1])
            lowest_low = np.min(low_12h[i-13:i+1])
            if atr_sum[i] > 0 and (highest_high - lowest_low) > 0:
                chop[i] = 100 * np.log10(atr_sum[i] / (highest_high - lowest_low)) / np.log10(14)
    
    # Calculate 1d volume moving average (20-period)
    volume_ma_20_1d = np.full(len(volume_1d), np.nan)
    for i in range(19, len(volume_1d)):
        if not np.isnan(volume_1d[i-19:i+1]).any():
            volume_ma_20_1d[i] = np.mean(volume_1d[i-19:i+1])
    
    # Align all HTF/LTF indicators to 12h timeframe
    alligator_jaw_aligned = align_htf_to_ltf(prices, df_1d, alligator_jaw)
    alligator_teeth_aligned = align_htf_to_ltf(prices, df_1d, alligator_teeth)
    alligator_lips_aligned = align_htf_to_ltf(prices, df_1d, alligator_lips)
    chop_aligned = align_htf_to_ltf(prices, prices, chop)
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(alligator_jaw_aligned[i]) or np.isnan(alligator_teeth_aligned[i]) or 
            np.isnan(alligator_lips_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(volume_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period MA
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        volume_confirm = volume_1d_aligned[i] > 1.5 * volume_ma_20_1d_aligned[i]
        
        # Chop regime filter: CHOP < 38.2 = trending, CHOP > 61.8 = ranging
        chop_trending = chop_aligned[i] < 38.2
        chop_ranging = chop_aligned[i] > 61.8
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price > Alligator Lips + volume confirmation + chop trending
            if close_12h[i] > alligator_lips_aligned[i] and volume_confirm and chop_trending:
                position = 1
                signals[i] = 0.25
            # Short entry: Price < Alligator Jaw + volume confirmation + chop trending
            elif close_12h[i] < alligator_jaw_aligned[i] and volume_confirm and chop_trending:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Price crosses back inside Alligator mouth OR chop regime shifts to ranging
            if position == 1:  # Long position
                if close_12h[i] < alligator_teeth_aligned[i] or chop_ranging:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if close_12h[i] > alligator_teeth_aligned[i] or chop_ranging:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals