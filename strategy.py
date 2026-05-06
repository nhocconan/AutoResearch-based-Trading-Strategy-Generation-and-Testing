#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d ATR-based breakout with 12h volume confirmation and 1d Choppiness regime filter
# - Uses 1d ATR(14) to calculate dynamic breakout bands (mean ± 1.5*ATR)
# - Uses 12h volume ratio to confirm institutional participation
# - Uses 1d Choppiness Index(14) to identify trending regimes (CHOP < 38.2) for breakout trades
# - Enters long when price breaks above mean + 1.5*ATR(1d) in trending regime with volume spike
# - Enters short when price breaks below mean - 1.5*ATR(1d) in trending regime with volume spike
# - Exits when price crosses back below/above the mean or Choppiness Index rises above 61.8 (range)
# - Designed to capture institutional breakouts in trending markets with volatility-adjusted sizing
# - Target: 100-200 total trades over 4 years (25-50/year) with 0.25 position sizing

name = "4h_1dATRBreakout_12hVolume_1dChoppium"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR and Choppiness Index calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Get 12h data for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Calculate 1d ATR(14)
    atr_period = 14
    tr1 = df_1d['high'].values - df_1d['low'].values
    tr2 = np.abs(df_1d['high'].values - np.roll(df_1d['close'].values, 1))
    tr3 = np.abs(df_1d['low'].values - np.roll(df_1d['close'].values, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Calculate 1d mean price (midpoint of range)
    mean_price = (df_1d['high'].values + df_1d['low'].values) / 2
    
    # Calculate dynamic breakout bands
    upper_band = mean_price + 1.5 * atr
    lower_band = mean_price - 1.5 * atr
    
    # Calculate 1d Choppiness Index(14)
    # CHOP = 100 * log10(sum(TR14) / (ATR14 * 14)) / log10(14)
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    atr14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    chop_raw = 100 * np.log10(tr_sum / (atr14 * 14)) / np.log10(14)
    
    # Align 1d indicators to 4h timeframe
    upper_band_4h = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_band_4h = align_htf_to_ltf(prices, df_1d, lower_band)
    mean_price_4h = align_htf_to_ltf(prices, df_1d, mean_price)
    chop_4h = align_htf_to_ltf(prices, df_1d, chop_raw)
    
    # Calculate 12h volume ratio (current volume / 20-period average)
    vol_ma_20_12h = pd.Series(df_12h['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio_12h = df_12h['volume'].values / vol_ma_20_12h
    vol_ratio_12h_4h = align_htf_to_ltf(prices, df_12h, vol_ratio_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN
        if (np.isnan(upper_band_4h[i]) or np.isnan(lower_band_4h[i]) or np.isnan(mean_price_4h[i]) or
            np.isnan(chop_4h[i]) or np.isnan(vol_ratio_12h_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Look for trending regime (CHOP < 38.2) and volume spike (vol ratio > 1.8)
            trending_regime = chop_4h[i] < 38.2
            volume_spike = vol_ratio_12h_4h[i] > 1.8
            
            if trending_regime and volume_spike:
                # Long: price breaks above upper band
                if close[i] > upper_band_4h[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below lower band
                elif close[i] < lower_band_4h[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price crosses below mean OR chop rises above 61.8 (range)
            if close[i] < mean_price_4h[i] or chop_4h[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above mean OR chop rises above 61.8 (range)
            if close[i] > mean_price_4h[i] or chop_4h[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals