#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot L3/H3 breakout with 12h volume confirmation and ADX trend filter
# - Primary signal: Close breaks above H3 (long) or below L3 (short) from prior day's Camarilla levels
# - Volume confirmation: 12h volume > 20-period median volume (avoid low-participation breakouts)
# - Trend filter: 12h ADX > 25 ensures breakout occurs in trending market (reduces false signals in chop)
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 19-50 trades/year (75-200 total over 4 years) per 4h strategy guidelines
# - Works in bull/bear: Camarilla levels provide structure, ADX filter ensures momentum, volume confirms participation

name = "4h_12h_camarilla_breakout_adx_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Pre-compute 12h indicators
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # 12h ADX(14) for trend strength
    # +DI, -DI, DX calculation
    up = np.diff(high_12h, prepend=high_12h[0])
    down = np.diff(low_12h, prepend=low_12h[0]) * -1  # make positive
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    
    # True Range
    tr1 = np.abs(np.diff(high_12h, prepend=high_12h[0]))
    tr2 = np.abs(np.diff(low_12h, prepend=low_12h[0]))
    tr3 = np.abs(np.diff(close_12h, prepend=close_12h[0]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smoothing (Wilder's smoothing = EMA with alpha=1/period)
    atr_12h = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di_12h = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_12h
    minus_di_12h = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_12h
    dx_12h = 100 * np.abs(plus_di_12h - minus_di_12h) / (plus_di_12h + minus_di_12h)
    adx_12h = pd.Series(dx_12h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # 12h volume regime: volume > 20-period median volume
    median_volume_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).median().values
    volume_regime_12h = volume_12h > median_volume_20
    
    # Align 12h indicators to 4h timeframe (completed 12h bar only)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    volume_regime_aligned = align_htf_to_ltf(prices, df_12h, volume_regime_12h.astype(float))
    
    # 4h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate Camarilla levels from prior 12h bar (use previous completed 12h bar)
    # Camarilla: H3 = C + (H-L)*1.1/4, L3 = C - (H-L)*1.1/4
    # We need to align the prior 12h bar's H, L, C to current 4h bar
    # Use shift=1 to get previous completed 12h bar
    df_12h_shifted = df_12h.copy()
    df_12h_shifted['high'] = np.roll(df_12h_shifted['high'], 1)
    df_12h_shifted['low'] = np.roll(df_12h_shifted['low'], 1)
    df_12h_shifted['close'] = np.roll(df_12h_shifted['close'], 1)
    # Set first value to NaN (no prior bar)
    df_12h_shifted.iloc[0] = np.nan
    
    # Calculate Camarilla levels from prior 12h bar
    camarilla_high = df_12h_shifted['high'].values
    camarilla_low = df_12h_shifted['low'].values
    camarilla_close = df_12h_shifted['close'].values
    
    camarilla_range = camarilla_high - camarilla_low
    camarilla_h3 = camarilla_close + camarilla_range * 1.1 / 4
    camarilla_l3 = camarilla_close - camarilla_range * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_12h_shifted, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_12h_shifted, camarilla_l3)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_12h_aligned[i]) or
            np.isnan(volume_regime_aligned[i]) or
            np.isnan(camarilla_h3_aligned[i]) or
            np.isnan(camarilla_l3_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below L3 OR ADX drops below 20 (trend weakening)
            if close[i] < camarilla_l3_aligned[i] or adx_12h_aligned[i] < 20.0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above H3 OR ADX drops below 20 (trend weakening)
            if close[i] > camarilla_h3_aligned[i] or adx_12h_aligned[i] < 20.0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla breakout with volume confirmation and ADX filter
            # Long: close breaks above H3 AND volume regime AND ADX > 25
            if close[i] > camarilla_h3_aligned[i] and volume_regime_aligned[i] and adx_12h_aligned[i] > 25.0:
                position = 1
                signals[i] = 0.25
            # Short: close breaks below L3 AND volume regime AND ADX > 25
            elif close[i] < camarilla_l3_aligned[i] and volume_regime_aligned[i] and adx_12h_aligned[i] > 25.0:
                position = -1
                signals[i] = -0.25
    
    return signals