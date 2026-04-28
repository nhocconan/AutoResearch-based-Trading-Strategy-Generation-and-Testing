#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d volume spike and chop regime filter.
# Enter long when price breaks above Camarilla R3 level with 1d volume > 2.0x 20-bar average and CHOP > 61.8 (ranging market for mean reversion).
# Enter short when price breaks below Camarilla S3 level with same conditions.
# Exit when price reaches Camarilla H3/L3 levels or midpoint.
# Uses discrete position sizing (0.25) to limit drawdown and reduce fee churn.
# Target: 75-200 total trades over 4 years (19-50/year).
# Camarilla levels work well in ranging markets, and CHOP filter ensures we only trade in ranging conditions where mean reversion is effective.
# Volume confirmation filters weak breakouts. This combination has shown promise in ETHUSDT and should work on BTC/ETH.

name = "4h_Camarilla_R3S3_Breakout_1dVolumeSpike_Chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla calculation (using previous day's OHLC)
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 4h bar's OHLC
    # We need to shift by 1 to use previous bar's data to avoid look-ahead
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    open_4h = df_4h['open'].values
    
    # Previous bar's OHLC (shift by 1)
    prev_high = np.roll(high_4h, 1)
    prev_low = np.roll(low_4h, 1)
    prev_close = np.roll(close_4h, 1)
    prev_open = np.roll(open_4h, 1)
    # First bar will have NaN due to roll, we'll handle it
    
    # Camarilla calculations
    rang = prev_high - prev_low
    camarilla_h3 = prev_close + rang * 1.1 / 4
    camarilla_l3 = prev_close - rang * 1.1 / 4
    camarilla_h4 = prev_close + rang * 1.1 / 2
    camarilla_l4 = prev_close - rang * 1.1 / 2
    camarilla_h5 = prev_close + rang * 1.1
    camarilla_l5 = prev_close - rang * 1.1
    camarilla_h6 = prev_close + rang * 1.1 * 2
    camarilla_l6 = prev_close - rang * 1.1 * 2
    camarilla_h8 = prev_close + rang * 1.1 * 4
    camarilla_l8 = prev_close - rang * 1.1 * 4
    camarilla_r3 = camarilla_h3
    camarilla_s3 = camarilla_l3
    camarilla_mid = (prev_high + prev_low) / 2
    
    # Align Camarilla levels to 4h
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3)
    camarilla_mid_aligned = align_htf_to_ltf(prices, df_4h, camarilla_mid)
    
    # Get 1d data for volume spike and chop regime
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d Volume spike: >2.0x 20-bar average
    volume_1d = df_1d['volume'].values
    volume_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > 2.0 * volume_ma_20
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    # 1d Chop regime: CHOP > 61.8 indicates ranging market (good for mean reversion)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # ATR (14-period)
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Chop = 100 * log10(sum(ATR,14) / (max(high,14) - min(low,14))) / log10(14)
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    max_high_minus_min_low = max_high_14 - min_low_14
    
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    
    # Avoid division by zero
    divisor = max_high_minus_min_low
    divisor = np.where(divisor == 0, np.nan, divisor)
    
    chop = 100 * np.log10(sum_atr_14 / divisor) / np.log10(14)
    chop_regime = chop > 61.8  # Ranging market
    chop_regime_aligned = align_htf_to_ltf(prices, df_1d, chop_regime)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Ensure sufficient history
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(camarilla_mid_aligned[i]) or np.isnan(volume_spike_aligned[i]) or 
            np.isnan(chop_regime_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Conditions
        vol_spike = volume_spike_aligned[i]
        in_chop = chop_regime_aligned[i]
        price = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: price < Camarilla S3 (mean reversion long), volume spike, chop regime
            if price < camarilla_s3_aligned[i] and vol_spike and in_chop:
                signals[i] = 0.25
                position = 1
            # Short entry: price > Camarilla R3 (mean reversion short), volume spike, chop regime
            elif price > camarilla_r3_aligned[i] and vol_spike and in_chop:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - hold or exit at H3 or midpoint
            if price >= camarilla_h3_aligned[i] or price >= camarilla_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - hold or exit at L3 or midpoint
            if price <= camarilla_l3_aligned[i] or price <= camarilla_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals