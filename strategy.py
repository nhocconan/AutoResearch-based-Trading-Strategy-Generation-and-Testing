#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 1d regime filter (ADX) and volume confirmation
# - Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
# - Bull regime: 1d ADX > 25 AND 1d +DI > -DI (strong uptrend)
# - Bear regime: 1d ADX > 25 AND 1d -DI > +DI (strong downtrend)
# - Range regime: 1d ADX < 20 (choppy market)
# - Long signals: Bull regime AND Bull Power > 0 AND volume > 1.5x 20-period average
# - Short signals: Bear regime AND Bear Power < 0 AND volume > 1.5x 20-period average
# - Exit: Opposite Elder Ray signal or regime change to range
# - Uses 6h timeframe for entries, 1d for regime/trend filter
# - Target: 12-37 trades/year on 6h (50-150 total over 4 years) to avoid fee drag

name = "6h_1d_elderray_volume_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d EMA(13) for Elder Ray
    close_1d = df_1d['close'].values
    ema_13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Pre-compute 1d ADX(14) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_adx = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d_adx[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d_adx[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Wilder smoothing (14-period)
    tr_14 = np.full_like(tr, np.nan, dtype=float)
    dm_plus_14 = np.full_like(dm_plus, np.nan, dtype=float)
    dm_minus_14 = np.full_like(dm_minus, np.nan, dtype=float)
    
    if len(tr) >= 14:
        tr_14[13] = np.nanmean(tr[1:14])
        dm_plus_14[13] = np.nanmean(dm_plus[1:14])
        dm_minus_14[13] = np.nanmean(dm_minus[1:14])
        
        for i in range(14, len(tr)):
            tr_14[i] = tr_14[i-1] - (tr_14[i-1] / 14) + tr[i]
            dm_plus_14[i] = dm_plus_14[i-1] - (dm_plus_14[i-1] / 14) + dm_plus[i]
            dm_minus_14[i] = dm_minus_14[i-1] - (dm_minus_14[i-1] / 14) + dm_minus[i]
    
    di_plus = np.full_like(tr_14, np.nan, dtype=float)
    di_minus = np.full_like(tr_14, np.nan, dtype=float)
    mask = ~np.isnan(tr_14) & (tr_14 != 0)
    di_plus[mask] = (dm_plus_14[mask] / tr_14[mask]) * 100
    di_minus[mask] = (dm_minus_14[mask] / tr_14[mask]) * 100
    
    dx = np.full_like(di_plus, np.nan, dtype=float)
    mask_dx = (~np.isnan(di_plus) & ~np.isnan(di_minus) & ((di_plus + di_minus) != 0))
    dx[mask_dx] = (np.abs(di_plus[mask_dx] - di_minus[mask_dx]) / (di_plus[mask_dx] + di_minus[mask_dx])) * 100
    
    adx_1d = np.full_like(dx, np.nan, dtype=float)
    if len(dx) >= 14:
        valid_dx = dx[14:28]
        if not np.all(np.isnan(valid_dx)):
            adx_1d[27] = np.nanmean(valid_dx)
            for i in range(28, len(dx)):
                if not np.isnan(dx[i]) and not np.isnan(adx_1d[i-1]):
                    adx_1d[i] = (adx_1d[i-1] * 13 + dx[i]) / 14
    
    # Align HTF indicators to 6h timeframe
    ema_13_aligned = align_htf_to_ltf(prices, df_1d, ema_13)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    di_plus_aligned = align_htf_to_ltf(prices, df_1d, di_plus)
    di_minus_aligned = align_htf_to_ltf(prices, df_1d, di_minus)
    
    # Pre-compute 6h Elder Ray components
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    bull_power = high_6h - ema_13_aligned  # High - EMA(13)
    bear_power = low_6h - ema_13_aligned   # Low - EMA(13)
    
    # Pre-compute 6h volume MA(20)
    vol_6h = prices['volume'].values
    vol_ma_20 = np.full_like(vol_6h, np.nan, dtype=float)
    for i in range(19, len(vol_6h)):
        vol_ma_20[i] = np.mean(vol_6h[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(di_plus_aligned[i]) or 
            np.isnan(di_minus_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Determine 1d regime
        adx_now = adx_1d_aligned[i]
        di_plus_now = di_plus_aligned[i]
        di_minus_now = di_minus_aligned[i]
        
        is_bull_regime = (adx_now > 25) and (di_plus_now > di_minus_now)
        is_bear_regime = (adx_now > 25) and (di_minus_now > di_plus_now)
        is_range_regime = adx_now < 20
        
        # Volume spike condition (1.5x average)
        vol_spike = vol_6h[i] > 1.5 * vol_ma_20[i]
        
        # Elder Ray signals
        bull_signal = bull_power[i] > 0  # Positive bull power
        bear_signal = bear_power[i] < 0  # Negative bear power
        
        close_now = close_6h[i]
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: bull regime AND bull power positive AND volume spike
            if (is_bull_regime and bull_signal and vol_spike):
                position = 1
                signals[i] = 0.25
            # Short conditions: bear regime AND bear power negative AND volume spike
            elif (is_bear_regime and bear_signal and vol_spike):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: opposite Elder Ray signal or regime change to range
            exit_long = (position == 1 and 
                        (not bull_signal or is_range_regime or is_bear_regime))
            exit_short = (position == -1 and 
                         (not bear_signal or is_range_regime or is_bull_regime))
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals