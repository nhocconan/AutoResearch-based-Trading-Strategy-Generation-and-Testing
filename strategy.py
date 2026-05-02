#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d volume spike and chop regime filter
# Camarilla R3/S3 levels act as strong intraday support/resistance derived from 1d OHLC
# Breakout above R3 or below S3 with 1d volume > 1.5x 20-period EMA indicates institutional participation
# Choppiness Index (CHOP) > 61.8 on 1d filters range markets to avoid false breakouts
# Designed for 12h timeframe targeting 12-37 trades/year (50-150 total over 4 years)
# Uses discrete position sizing (0.25) to minimize fee churn and control drawdown
# Works in trending markets (breakouts with volume) and avoids ranging markets via CHOP filter

name = "12h_Camarilla_R3S3_Breakout_1dVolume_Chop"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for volume confirmation and chop regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for CHOP calculation
        return np.zeros(n)
    
    # 1d Volume EMA for confirmation
    vol_ema_20 = pd.Series(df_1d['volume']).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmation = df_1d['volume'].values > (1.5 * vol_ema_20)
    volume_confirmation_aligned = align_htf_to_ltf(prices, df_1d, volume_confirmation)
    
    # 1d Choppiness Index (CHOP) - range/trend regime filter
    # True Range
    tr1 = pd.Series(df_1d['high']).diff().abs()
    tr2 = pd.Series(df_1d['low']).diff().abs()
    tr3 = (pd.Series(df_1d['close']).shift() - pd.Series(df_1d['high'])).abs()
    tr4 = (pd.Series(df_1d['close']).shift() - pd.Series(df_1d['low'])).abs()
    tr = pd.concat([tr1, tr2, tr3, tr4], axis=1).max(axis=1)
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max()
    ll = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min()
    
    # Choppiness Index: CHOP = 100 * log10(atr_sum / (hh - ll)) / log10(14)
    # Range: CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    chop = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(14)
    chop_values = chop.values
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_values)
    
    # Calculate Camarilla levels for each 1d bar (based on same day's OHLC)
    # Standard Camarilla: R3 = close + (high-low)*1.1/4, S3 = close - (high-low)*1.1/4
    camarilla_r3 = df_1d['close'].values + (df_1d['high'].values - df_1d['low'].values) * 1.1 / 4
    camarilla_s3 = df_1d['close'].values - (df_1d['high'].values - df_1d['low'].values) * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe (use same day's levels)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(volume_confirmation_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade in trending markets (CHOP < 61.8)
        trending_market = chop_aligned[i] < 61.8
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout above R3 with volume confirmation and trending market
            if close[i] > camarilla_r3_aligned[i] and volume_confirmation_aligned[i] and trending_market:
                signals[i] = 0.25
                position = 1
            # Short: Breakout below S3 with volume confirmation and trending market
            elif close[i] < camarilla_s3_aligned[i] and volume_confirmation_aligned[i] and trending_market:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below S3 (reversal) OR market becomes ranging
            if close[i] < camarilla_s3_aligned[i] or chop_aligned[i] >= 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price breaks above R3 (reversal) OR market becomes ranging
            if close[i] > camarilla_r3_aligned[i] or chop_aligned[i] >= 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals