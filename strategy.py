#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Camarilla pivot breakout with volume confirmation and chop regime filter.
# Uses weekly Camarilla R3/S3 levels for entry timing, filtered by daily EMA200 trend and volume spike.
# Chop regime filter (Choppiness Index > 61.8) avoids whipsaws in ranging markets.
# Designed for low trade frequency (10-20/year) to minimize fee drag. Works in bull/bear: 
# - Bull market: EMA200 filter ensures long bias alignment
# - Bear market: Weekly S3 breakdown captures institutional selling pressure with volume confirmation
# - Chop filter prevents false breakouts during consolidation

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours to avoid datetime operations in loop
    hours = pd.DatetimeIndex(open_time).hour
    
    # Get 1d and 1w HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 20:
        return np.zeros(n)
    
    # === 1d Indicators ===
    # Daily EMA(200) for trend bias
    ema_200_1d = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Daily ATR(14) for stoploss and volume normalization
    atr_14_1d = pd.Series(np.abs(df_1d['high'].values - df_1d['low'].values)).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Daily volume SMA(20) for volume confirmation
    vol_sma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_20_1d)
    
    # === 1w Indicators: Camarilla Pivots (R3, S3) ===
    # Camarilla formula: Close +- (High-Low) * 1.1/4
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    camarilla_high_1w = close_1w + (high_1w - low_1w) * 1.1 / 4  # R3 level
    camarilla_low_1w = close_1w - (high_1w - low_1w) * 1.1 / 4    # S3 level
    
    camarilla_high_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_high_1w)
    camarilla_low_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_low_1w)
    
    # === Chop Regime Filter (Daily) ===
    # Choppiness Index: measures whether market is choppy (ranging) or trending
    # CHOP > 61.8 = ranging/choppy market (avoid breakout trades)
    # CHOP < 38.2 = trending market (favor breakout trades)
    true_range = np.maximum(
        np.maximum(
            np.abs(df_1d['high'].values - df_1d['low'].values),
            np.abs(df_1d['high'].values - np.roll(df_1d['close'].values, 1))
        ),
        np.abs(np.roll(df_1d['close'].values, 1) - df_1d['low'].values)
    )
    # Set first TR to high-low to avoid roll artifact
    true_range[0] = np.abs(df_1d['high'].values[0] - df_1d['low'].values[0])
    
    atr_14_chop = pd.Series(true_range).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    chop_denominator = np.where((highest_high_14 - lowest_low_14) == 0, 1, (highest_high_14 - lowest_low_14))
    chop_value = 100 * np.log10(atr_14_chop / chop_denominator * np.sqrt(14)) / np.log10(14)
    chop_value_aligned = align_htf_to_ltf(prices, df_1d, chop_value)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 200
    
    for i in range(warmup, n):
        # Session filter: 08-20 UTC only (reduces noise trades)
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
            
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20_1d_aligned[i] * 1.5)
        
        # Chop regime filter: only trade when market is trending (CHOP < 38.2)
        chop_filter = chop_value_aligned[i] < 38.2
        
        # Skip if any required data is NaN
        if (np.isnan(camarilla_high_1w_aligned[i]) or np.isnan(camarilla_low_1w_aligned[i]) or
            np.isnan(ema_200_1d_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or
            np.isnan(chop_value_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Weekly Camarilla R3 (strong resistance)
        # 2. Daily price above EMA200 (bullish trend bias)
        # 3. Volume confirmation
        # 4. Trending market regime (low chop)
        if (close[i] > camarilla_high_1w_aligned[i] and
            close[i] > ema_200_1d_aligned[i] and
            vol_confirm and
            chop_filter):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Weekly Camarilla S3 (strong support)
        # 2. Daily price below EMA200 (bearish trend bias)
        # 3. Volume confirmation
        # 4. Trending market regime (low chop)
        elif (close[i] < camarilla_low_1w_aligned[i] and
              close[i] < ema_200_1d_aligned[i] and
              vol_confirm and
              chop_filter):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1d_Camarilla_R3S3_EMA200_VolChopFilter_v1"
timeframe = "1d"
leverage = 1.0