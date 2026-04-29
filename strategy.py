#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h TRIX + Volume Spike + Choppiness Regime Filter
# TRIX (Triple Exponential Average) identifies momentum and trend changes with less lag
# Long when TRIX crosses above zero + volume > 2.0x 20-period average + choppy market (CHOP > 61.8)
# Short when TRIX crosses below zero + volume > 2.0x 20-period average + choppy market (CHOP > 61.8)
# Choppy market filter prevents whipsaws in strong trends and works in both bull/bear regimes
# Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe

name = "4h_TRIX_VolumeSpike_ChopFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter (aligned)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate TRIX (15,9,9) on 4h data
    # TRIX = EMA(EMA(EMA(close, 15), 9), 9) - 1 period ago
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema3 = pd.Series(ema2).ewm(span=9, adjust=False, min_periods=9).mean().values
    trix = np.diff(ema3, prepend=ema3[0]) / ema3[:-1] * 100 if len(ema3) > 1 else np.zeros_like(ema3)
    trix = np.append(trix, 0.0)  # same length as close
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Choppiness Index (CHOP) on 1d data for regime filter
    # CHOP = 100 * log10(sum(ATR(1)) / (max(high) - min(low))) / log10(n)
    # Simplified: CHOP > 61.8 = ranging/choppy, CHOP < 38.2 = trending
    atr_1d = np.zeros(len(df_1d))
    for i in range(1, len(df_1d)):
        tr = max(
            df_1d['high'].iloc[i] - df_1d['low'].iloc[i],
            abs(df_1d['high'].iloc[i] - df_1d['close'].iloc[i-1]),
            abs(df_1d['low'].iloc[i] - df_1d['close'].iloc[i-1])
        )
        atr_1d[i] = tr if i == 1 else (atr_1d[i-1] * 13 + tr) / 14  # Wilder's ATR
    
    chop_raw = np.zeros(len(df_1d))
    lookback = 14
    for i in range(lookback, len(df_1d)):
        atr_sum = np.sum(atr_1d[i-lookback+1:i+1])
        max_high = np.max(df_1d['high'].iloc[i-lookback+1:i+1])
        min_low = np.min(df_1d['low'].iloc[i-lookback+1:i+1])
        if max_high > min_low:
            chop_raw[i] = 100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(lookback)
        else:
            chop_raw[i] = 50.0  # neutral when no range
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_raw)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(15+9+9, 34, 20, 14)  # TRIX warmup, 1d EMA34, volume MA, CHOP
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(trix[i]) or np.isnan(chop_aligned[i]) or i == 0):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_trix = trix[i]
        prev_trix = trix[i-1]
        curr_ema_1d = ema_34_1d_aligned[i]
        curr_volume = volume[i]
        curr_vol_ma = vol_ma_20[i]
        curr_chop = chop_aligned[i]
        
        # Volume confirmation: current volume > 2.0x 20-period average
        vol_confirm = curr_volume > 2.0 * curr_vol_ma
        
        # Choppy market regime: CHOP > 61.8 = ranging (good for mean reversion/breakouts)
        chop_regime = curr_chop > 61.8
        
        # Handle exits and trailing logic
        if position == 1:  # Long position
            # Exit: TRIX crosses below zero (momentum fading)
            if curr_trix < 0 and prev_trix >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: TRIX crosses above zero (momentum fading)
            if curr_trix > 0 and prev_trix <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: TRIX crosses above zero + volume confirmation + choppy regime
            if (i > start_idx and 
                curr_trix > 0 and prev_trix <= 0 and  # TRIX cross above zero
                vol_confirm and 
                chop_regime):
                signals[i] = 0.25
                position = 1
            # Short entry: TRIX crosses below zero + volume confirmation + choppy regime
            elif (i > start_idx and 
                  curr_trix < 0 and prev_trix >= 0 and  # TRIX cross below zero
                  vol_confirm and 
                  chop_regime):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals