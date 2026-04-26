#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeSpike_RegimeFilter
Hypothesis: Camarilla R1/S1 breakout with 1d EMA34 trend filter, volume spike confirmation, and chop regime filter to avoid whipsaws in sideways markets. 
Designed for low trade frequency (~25-35/year) to minimize fee drag while capturing strong trending moves in both bull and bear markets. 
Uses discrete position sizing (0.30) for balanced risk/reward. Focus on BTC/ETH with SOL as secondary confirmation.
Adding chop regime filter (CHOP > 61.8 = range, avoid entries) to improve performance in bear/ranging markets like 2025.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels and EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].values
    
    # Camarilla R1 and S1 levels (tighter breakout levels for higher quality signals)
    R1 = close_1d_prev + (high_1d - low_1d) * 1.1 / 12
    S1 = close_1d_prev - (high_1d - low_1d) * 1.1 / 12
    
    # Align Camarilla levels
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Volume confirmation: 2.5x average volume (stricter to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stoploss (using 14-period ATR)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Choppiness Index regime filter (using 14-period)
    # CHOP > 61.8 = ranging market (avoid entries), CHOP < 38.2 = trending market (favor entries)
    hl_range = np.maximum(high, low) - np.minimum(high, low)
    chop_numerator = np.log10(pd.Series(hl_range).rolling(window=14, min_periods=14).sum().values)
    chop_denominator = np.log10(14) + np.log10(pd.Series(atr).rolling(window=14, min_periods=14).sum().values)
    chop = 100 * chop_numerator / chop_denominator
    chop_regime = chop < 61.8  # True when trending (CHOP < 61.8), False when ranging
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of 1d EMA (34), volume MA (20), ATR (14), CHOP (14)
    start_idx = max(34, 20, 14, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(chop_regime[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.30
            else:
                signals[i] = -0.30
            continue
        
        ema_34_1d_val = ema_34_1d_aligned[i]
        R1_val = R1_aligned[i]
        S1_val = S1_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_ma_val = vol_ma[i]
        atr_val = atr[i]
        chop_regime_val = chop_regime[i]
        
        # Only allow entries in trending regimes (avoid whipsaws in ranging markets)
        if not chop_regime_val:
            # In ranging market, hold current position or go flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.30
            else:
                signals[i] = -0.30
            continue
        
        if position == 0:
            # Long: price breaks above R1 with volume confirmation and uptrend
            long_signal = (high_val > R1_val) and (volume_val > 2.5 * vol_ma_val) and (close_val > ema_34_1d_val)
            # Short: price breaks below S1 with volume confirmation and downtrend
            short_signal = (low_val < S1_val) and (volume_val > 2.5 * vol_ma_val) and (close_val < ema_34_1d_val)
            
            if long_signal:
                signals[i] = 0.30
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.30
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.30
            # Exit: ATR stoploss or trend reversal
            if (close_val < entry_price - 2.5 * atr_val or 
                close_val < ema_34_1d_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.30
            # Exit: ATR stoploss or trend reversal
            if (close_val > entry_price + 2.5 * atr_val or 
                close_val > ema_34_1d_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeSpike_RegimeFilter"
timeframe = "4h"
leverage = 1.0