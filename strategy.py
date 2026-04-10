#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d trend filter and volume confirmation
# - Williams Alligator: Jaw (13-period SMMA, 8-bar shift), Teeth (8-period SMMA, 5-bar shift), Lips (5-period SMMA, 3-bar shift)
# - Long when Lips > Teeth > Jaw (bullish alignment) with 1d uptrend (close > EMA50) and volume spike
# - Short when Lips < Teeth < Jaw (bearish alignment) with 1d downtrend (close < EMA50) and volume spike
# - Uses 6h timeframe targeting 12-37 trades/year (50-150 total over 4 years) to minimize fee drag
# - 1d EMA50 filter ensures trading with higher timeframe trend direction
# - 6h volume > 2.0x 20-period average confirms breakout strength
# - Discrete position sizing (0.25) to minimize fee churn
# - ATR-based stoploss: exit when price moves against position by 2.0x ATR(14) or Alligator lines cross

name = "6h_1d_alligator_volume_trend_atr_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d volume confirmation: > 2.0x 20-period average
    avg_volume_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (2.0 * avg_volume_20_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # 1d ATR(14) for stoploss
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14_1d = np.zeros_like(tr)
    atr_14_1d[14-1] = np.mean(tr[:14])
    for i in range(14, len(tr)):
        atr_14_1d[i] = (atr_14_1d[i-1] * (14-1) + tr[i]) / 14
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # 6h Williams Alligator
    close_6h = prices['close'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    
    # Jaw: 13-period SMMA of median price, shifted by 8 bars
    median_price = (high_6h + low_6h) / 2
    jaw_raw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    jaw = np.roll(jaw_raw, 8)
    jaw[:8] = np.nan  # First 8 values invalid due to shift
    
    # Teeth: 8-period SMMA of median price, shifted by 5 bars
    teeth_raw = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values
    teeth = np.roll(teeth_raw, 5)
    teeth[:5] = np.nan  # First 5 values invalid due to shift
    
    # Lips: 5-period SMMA of median price, shifted by 3 bars
    lips_raw = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values
    lips = np.roll(lips_raw, 3)
    lips[:3] = np.nan  # First 3 values invalid due to shift
    
    # Alligator alignment signals
    bullish_alignment = (lips > teeth) & (teeth > jaw)
    bearish_alignment = (lips < teeth) & (teeth < jaw)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    entry_atr = 0.0
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_spike_1d_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: ATR-based stoploss or Alligator bearish crossover
            if (prices['close'].iloc[i] < entry_price - 2.0 * entry_atr or 
                (lips[i] < teeth[i] and teeth[i] < jaw[i])):  # Bearish alignment
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: ATR-based stoploss or Alligator bullish crossover
            if (prices['close'].iloc[i] > entry_price + 2.0 * entry_atr or 
                (lips[i] > teeth[i] and teeth[i] > jaw[i])):  # Bullish alignment
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Alligator alignment with trend and volume filters
            if vol_spike_1d_aligned[i]:
                # Long signal: bullish Alligator alignment in 1d uptrend
                if bullish_alignment[i] and prices['close'].iloc[i] > ema_50_1d_aligned[i]:
                    position = 1
                    entry_price = prices['close'].iloc[i]
                    entry_atr = atr_14_1d_aligned[i]
                    signals[i] = 0.25
                # Short signal: bearish Alligator alignment in 1d downtrend
                elif bearish_alignment[i] and prices['close'].iloc[i] < ema_50_1d_aligned[i]:
                    position = -1
                    entry_price = prices['close'].iloc[i]
                    entry_atr = atr_14_1d_aligned[i]
                    signals[i] = -0.25
    
    return signals