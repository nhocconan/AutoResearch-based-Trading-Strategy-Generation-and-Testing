#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout + 1d volume spike + choppiness regime filter
# Camarilla pivot levels provide precise support/resistance for institutional breakouts,
# volume spike confirms participation, choppiness regime ensures we trade with the trend.
# Designed to work in both bull and bear markets by adapting to volatility regimes.
# Target: 20-50 trades/year (80-200 over 4 years).

name = "4h_Camarilla_R3S3_Breakout_1dVolumeSpike_ChopRegime"
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
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for volume spike and choppiness
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d volume spike (volume > 2.0 * 20-period EMA of volume)
    vol_ema_20 = pd.Series(df_1d['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = df_1d['volume'].values > (2.0 * vol_ema_20)
    
    # Calculate 1d choppiness index (CHOP)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR(14) - sum of TR over 14 periods
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(atr_14 / (hh_14 - ll_14)) / log10(14)
    range_14 = hh_14 - ll_14
    chop = 100 * np.log10(atr_14 / range_14) / np.log10(14)
    # Handle division by zero and invalid values
    chop = np.where((range_14 == 0) | np.isnan(chop), 50.0, chop)
    
    # Align 1d indicators to 4h timeframe
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate Camarilla pivot levels from previous 1d
    # Typical price = (H + L + C) / 3
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    typical_price_vals = typical_price.values
    
    # Camarilla levels: R3/S3 = typical_price ± 1.1 * (H - L) / 2
    hl_range = df_1d['high'].values - df_1d['low'].values
    camarilla_r3 = typical_price_vals + 1.1 * hl_range / 2
    camarilla_s3 = typical_price_vals - 1.1 * hl_range / 2
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(volume_spike_aligned[i]) or np.isnan(chop_aligned[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: CHOP < 38.2 = trending (breakout), CHOP > 61.8 = ranging (mean revert)
        is_trending = chop_aligned[i] < 38.2
        is_ranging = chop_aligned[i] > 61.8
        
        if position == 0:
            # Long: Close above R3 + volume spike + (trending OR ranging)
            if close[i] > camarilla_r3_aligned[i] and volume_spike_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close below S3 + volume spike + (trending OR ranging)
            elif close[i] < camarilla_s3_aligned[i] and volume_spike_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Close below midpoint of R3 and S3 OR reverse signal
            midpoint = (camarilla_r3_aligned[i] + camarilla_s3_aligned[i]) / 2
            if close[i] < midpoint or (close[i] < camarilla_s3_aligned[i] and volume_spike_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Close above midpoint of R3 and S3 OR reverse signal
            midpoint = (camarilla_r3_aligned[i] + camarilla_s3_aligned[i]) / 2
            if close[i] > midpoint or (close[i] > camarilla_r3_aligned[i] and volume_spike_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals