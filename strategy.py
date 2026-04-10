#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d trend filter + volume confirmation
# - Williams Alligator (13,8,5 SMAs with 8,5,3 offsets): long when Jaw < Teeth < Lips (bullish alignment)
# - Short when Jaw > Teeth > Lips (bearish alignment)
# - 1d ADX(14) > 20 to ensure sufficient trend strength and avoid choppy markets
# - Volume confirmation: current 12h volume > 1.5x 20-period average
# - ATR-based stop: exit when price moves against position by 2.0*ATR(14)
# - Designed for 12h timeframe: targets 12-37 trades/year to minimize fee drag
# - Works in bull/bear markets: Alligator catches trends, ADX filter avoids false signals in ranging markets
# - Uses discrete position sizing (0.25) to minimize fee churn

name = "12h_1d_williams_alligator_adx_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d ADX(14) for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Pre-compute 12h Williams Alligator
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    # Jaw: 13-period SMA of median price, offset by 8 bars
    median_price = (high_12h + low_12h) / 2
    jaw_raw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw_raw, 8)
    jaw[:8] = np.nan
    
    # Teeth: 8-period SMA of median price, offset by 5 bars
    teeth_raw = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth_raw, 5)
    teeth[:5] = np.nan
    
    # Lips: 5-period SMA of median price, offset by 3 bars
    lips_raw = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips_raw, 3)
    lips[:3] = np.nan
    
    # Alligator alignment signals
    bullish_alignment = (jaw < teeth) & (teeth < lips)
    bearish_alignment = (jaw > teeth) & (teeth > lips)
    
    # Pre-compute 12h volume confirmation
    volume_12h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_12h > (1.5 * avg_volume_20)
    
    # Pre-compute 12h ATR(14) for stoploss
    tr1_12h = high_12h - low_12h
    tr2_12h = np.abs(high_12h - np.roll(close_12h, 1))
    tr3_12h = np.abs(low_12h - np.roll(close_12h, 1))
    tr_12h = np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))
    tr_12h[0] = tr1_12h[0]
    atr_14 = pd.Series(tr_12h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(vol_spike[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: stoploss hit OR Alligator alignment breaks down
            if close_12h[i] < entry_price - 2.0 * atr_14[i] or not bullish_alignment[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: stoploss hit OR Alligator alignment breaks down
            if close_12h[i] > entry_price + 2.0 * atr_14[i] or not bearish_alignment[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Alligator alignment with trend and volume filters
            if vol_spike[i] and adx_aligned[i] > 20:
                if bullish_alignment[i]:
                    position = 1
                    entry_price = close_12h[i]
                    signals[i] = 0.25
                elif bearish_alignment[i]:
                    position = -1
                    entry_price = close_12h[i]
                    signals[i] = -0.25
    
    return signals