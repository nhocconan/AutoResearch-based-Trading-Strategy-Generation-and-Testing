#!/usr/bin/env python3
"""
12h_WilliamsAlligator_Trend_Filter
12h strategy using Williams Alligator for trend direction, volume confirmation, and Bollinger Bandwidth regime filter.
- Long: Price above Alligator teeth (Jaw<Teeth<Lips alignment) + volume > 1.3x 20-period avg + BBW > 30th percentile
- Short: Price below Alligator teeth (Lips<Teeth<Jaw alignment) + volume > 1.3x 20-period avg + BBW > 30th percentile
- Exit: Opposite Alligator alignment or BBW < 20th percentile (low volatility)
Williams Alligator: Jaw=SMA(13,8), Teeth=SMA(8,5), Lips=SMA(5,3)
Designed for ~15-25 trades/year per symbol (60-100 total over 4 years)
Works in trending markets (trend following) and avoids ranging markets via BBW filter
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_ta, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Williams Alligator and Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Williams Alligator components
    # Jaw: SMA(13) of median price, shifted 8 bars
    median_price_1d = (high_1d + low_1d) / 2
    jaw_raw = pd.Series(median_price_1d).rolling(window=13, min_periods=13).mean().values
    jaw_raw = np.roll(jaw_raw, 8)  # shift 8 bars forward
    jaw_raw[:8] = np.nan  # first 8 values invalid
    
    # Teeth: SMA(8) of median price, shifted 5 bars
    teeth_raw = pd.Series(median_price_1d).rolling(window=8, min_periods=8).mean().values
    teeth_raw = np.roll(teeth_raw, 5)  # shift 5 bars forward
    teeth_raw[:5] = np.nan  # first 5 values invalid
    
    # Lips: SMA(5) of median price, shifted 3 bars
    lips_raw = pd.Series(median_price_1d).rolling(window=5, min_periods=5).mean().values
    lips_raw = np.roll(lips_raw, 3)  # shift 3 bars forward
    lips_raw[:3] = np.nan  # first 3 values invalid
    
    # Align Alligator lines to 12h
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_raw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_raw)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_raw)
    
    # Bollinger Bandwidth (20, 2) for regime filter
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    bbw_raw = (upper_bb - lower_bb) / sma_20 * 100  # percentage bandwidth
    
    # Align BBW to 12h and calculate percentiles
    bbw_aligned = align_htf_to_ltf(prices, df_1d, bbw_raw)
    # Calculate 20th and 30th percentiles using expanding window
    bbw_series = pd.Series(bbw_aligned)
    bbw_p20 = bbw_series.expanding(min_periods=20).quantile(0.20).values
    bbw_p30 = bbw_series.expanding(min_periods=20).quantile(0.30).values
    
    # Volume confirmation: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bbw_aligned[i]) or np.isnan(bbw_p20[i]) or np.isnan(bbw_p30[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Alligator alignment conditions
        bullish_alignment = jaw_aligned[i] < teeth_aligned[i] < lips_aligned[i]
        bearish_alignment = lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i]
        
        # Price relative to teeth (middle line)
        price_above_teeth = close[i] > teeth_aligned[i]
        price_below_teeth = close[i] < teeth_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.3 * vol_ma_20[i]
        
        # Bollinger Bandwidth regime filter (avoid low volatility)
        high_volatility = bbw_aligned[i] > bbw_p30[i]  # above 30th percentile
        low_volatility = bbw_aligned[i] < bbw_p20[i]   # below 20th percentile
        
        if position == 0:
            # Long: bullish alignment + price above teeth + volume + high volatility
            if bullish_alignment and price_above_teeth and vol_confirm and high_volatility:
                signals[i] = 0.25
                position = 1
            # Short: bearish alignment + price below teeth + volume + high volatility
            elif bearish_alignment and price_below_teeth and vol_confirm and high_volatility:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: bearish alignment OR price below teeth OR low volatility
            if bearish_alignment or not price_above_teeth or low_volatility:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: bullish alignment OR price above teeth OR low volatility
            if bullish_alignment or not price_below_teeth or low_volatility:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_Trend_Filter"
timeframe = "12h"
leverage = 1.0