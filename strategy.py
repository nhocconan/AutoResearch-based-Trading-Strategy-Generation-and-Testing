#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R extreme + 1d volume confirmation + 12h ADX trend filter
# - Primary signal: Williams %R(14) crosses above -20 (overbought) for short or below -80 (oversold) for long
# - Volume confirmation: 1d volume > 1.3x 20-period average (avoid low-participation extremes)
# - Trend filter: 12h ADX(14) > 25 ensures we only take signals in trending markets
# - Works in bull/bear: In strong trends (ADX > 25), extremes often continue; we fade only when volume confirms exhaustion
# - Position size: 0.25 discrete level to minimize fee churn
# - Target: 20-50 trades/year (75-200 total over 4 years) per 4h strategy guidelines
# - ATR-based stoploss: exit when price moves against position by 2.0x ATR(14)

name = "4h_1d_12h_williams_volume_adx_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_12h = get_htf_data(prices, '12h')
    if len(df_1d) < 50 or len(df_12h) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d volume confirmation
    volume_1d = df_1d['volume'].values
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume_1d > (1.3 * avg_volume_20)
    volume_confirm_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm)
    
    # Pre-compute 12h ADX(14) for trend filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high_12h - np.roll(high_12h, 1)) > (np.roll(low_12h, 1) - low_12h),
                       np.maximum(high_12h - np.roll(high_12h, 1), 0), 0)
    dm_minus = np.where((np.roll(low_12h, 1) - low_12h) > (high_12h - np.roll(high_12h, 1)),
                        np.maximum(np.roll(low_12h, 1) - low_12h, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM-
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Pre-compute 4h Williams %R(14)
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    
    highest_high = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_4h) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid div by zero
    
    # Pre-compute 4h ATR(14) for stoploss
    tr_4h1 = high_4h - low_4h
    tr_4h2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr_4h3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr_4h = np.maximum(tr_4h1, np.maximum(tr_4h2, tr_4h3))
    tr_4h[0] = tr_4h1[0]
    atr_14 = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(volume_confirm_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Williams %R crosses above -50 (mean reversion) OR stoploss hit
            if williams_r[i] > -50 or close_4h[i] < entry_price - 2.0 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R crosses below -50 (mean reversion) OR stoploss hit
            if williams_r[i] < -50 or close_4h[i] > entry_price + 2.0 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Williams %R extremes with volume confirmation and ADX trend filter
            # Only take signals when ADX > 25 (trending market) and volume confirms
            if volume_confirm_aligned[i] and adx_aligned[i] > 25:
                if williams_r[i] < -80:  # oversold - potential long
                    position = 1
                    entry_price = close_4h[i]
                    signals[i] = 0.25
                elif williams_r[i] > -20:  # overbought - potential short
                    position = -1
                    entry_price = close_4h[i]
                    signals[i] = -0.25
    
    return signals