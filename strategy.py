#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1w trend filter and volume confirmation
# - Williams Alligator: Jaw(13,8), Teeth(8,5), Lips(5,3) SMAs on median price
# - 1w ADX(14) > 25 to ensure strong weekly trend alignment
# - Volume confirmation: current 12h volume > 1.8x 30-period average
# - Long when Lips > Teeth > Jaw (bullish alignment), Short when Lips < Teeth < Jaw (bearish)
# - Designed for 12h timeframe: targets 12-37 trades/year (50-150 total over 4 years) to avoid fee drag
# - Works in bull/bear markets: weekly ADX filter ensures we trade with higher timeframe trend
# - Uses discrete position sizing (0.25) to minimize fee churn

name = "12h_1w_alligator_adx_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute 1w ADX(25) for trend filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w), 
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)), 
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_25 = pd.Series(tr).ewm(span=25, adjust=False, min_periods=25).mean().values
    dm_plus_25 = pd.Series(dm_plus).ewm(span=25, adjust=False, min_periods=25).mean().values
    dm_minus_25 = pd.Series(dm_minus).ewm(span=25, adjust=False, min_periods=25).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_25 / tr_25
    di_minus = 100 * dm_minus_25 / tr_25
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(span=25, adjust=False, min_periods=25).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Pre-compute Williams Alligator on 12h prices
    median_price = (prices['high'].values + prices['low'].values) / 2.0
    
    # Jaw: 13-period SMA, shifted 8 bars
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw, 8)
    jaw[:8] = np.nan
    
    # Teeth: 8-period SMA, shifted 5 bars
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth, 5)
    teeth[:5] = np.nan
    
    # Lips: 5-period SMA, shifted 3 bars
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips, 3)
    lips[:3] = np.nan
    
    # Pre-compute 12h volume confirmation
    volume_12h = prices['volume'].values
    avg_volume_30 = pd.Series(volume_12h).rolling(window=30, min_periods=30).mean().values
    vol_spike = volume_12h > (1.8 * avg_volume_30)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_aligned[i]) or np.isnan(lips[i]) or np.isnan(teeth[i]) or 
            np.isnan(jaw[i]) or np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Alligator alignment breaks (Lips <= Teeth or Teeth <= Jaw)
            if lips[i] <= teeth[i] or teeth[i] <= jaw[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Alligator alignment breaks (Lips >= Teeth or Teeth >= Jaw)
            if lips[i] >= teeth[i] or teeth[i] >= jaw[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Alligator alignment with trend and volume filters
            if vol_spike[i] and adx_aligned[i] > 25:
                # Bullish alignment: Lips > Teeth > Jaw
                if lips[i] > teeth[i] and teeth[i] > jaw[i]:
                    position = 1
                    signals[i] = 0.25
                # Bearish alignment: Lips < Teeth < Jaw
                elif lips[i] < teeth[i] and teeth[i] < jaw[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals