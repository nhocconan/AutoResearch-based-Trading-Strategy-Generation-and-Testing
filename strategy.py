#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator + 1w trend filter + volume confirmation
# - Williams Alligator: Jaw(13,8), Teeth(8,5), Lips(5,3) SMAs on median price
# - Trend: Alligator lines aligned (Jaw > Teeth > Lips for uptrend, reverse for downtrend)
# - 1w ADX(14) > 25 to ensure strong trending environment, avoid chop
# - Volume confirmation: current volume > 2.0x 50-period average to filter low-quality breaks
# - Long: Alligator uptrend + ADX filter + volume spike + price > Lips
# - Short: Alligator downtrend + ADX filter + volume spike + price < Lips
# - Designed for 1d timeframe: targets 20-50 trades/year to avoid fee drag
# - Works in bull/bear markets: ADX filter ensures we trade only when 1w trend is strong

name = "1d_1w_alligator_adx_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute 1w ADX(25) for trend filter (higher threshold for stronger trends)
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
    
    # Williams Alligator on 1d data
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    close_1d = prices['close'].values
    
    # Median price = (high + low) / 2
    median_price = (high_1d + low_1d) / 2.0
    
    # Alligator lines: SMAs of median price
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    jaw = pd.Series(jaw).shift(8)  # 8-bar forward shift
    
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    teeth = pd.Series(teeth).shift(5)  # 5-bar forward shift
    
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    lips = pd.Series(lips).shift(3)  # 3-bar forward shift
    
    # Alligator trend conditions
    jaw_gt_teeth = jaw > teeth
    teeth_gt_lips = teeth > lips
    jaw_lt_teeth = jaw < teeth
    teeth_lt_lips = teeth < lips
    
    # Pre-compute 1d volume confirmation
    volume_1d = prices['volume'].values
    avg_volume_50 = pd.Series(volume_1d).rolling(window=50, min_periods=50).mean().values
    vol_spike = volume_1d > (2.0 * avg_volume_50)
    
    # Pre-compute 1d ATR(14) for stoploss
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    tr_1d[0] = tr1_1d[0]
    atr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(50, n):  # Start after warmup period for Alligator
        # Skip if any required data is invalid
        if (np.isnan(adx_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(vol_spike[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Alligator trend reverses OR stoploss hit
            if not (jaw_gt_teeth[i] and teeth_gt_lips[i]) or close_1d[i] < entry_price - 2.5 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Alligator trend reverses OR stoploss hit
            if not (jaw_lt_teeth[i] and teeth_lt_lips[i]) or close_1d[i] > entry_price + 2.5 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Alligator signals with trend and volume filters
            if adx_aligned[i] > 25 and vol_spike[i]:
                # Uptrend: Jaw > Teeth > Lips
                if jaw_gt_teeth[i] and teeth_gt_lips[i] and close_1d[i] > lips[i]:
                    position = 1
                    entry_price = close_1d[i]
                    signals[i] = 0.25
                # Downtrend: Jaw < Teeth < Lips
                elif jaw_lt_teeth[i] and teeth_lt_lips[i] and close_1d[i] < lips[i]:
                    position = -1
                    entry_price = close_1d[i]
                    signals[i] = -0.25
    
    return signals