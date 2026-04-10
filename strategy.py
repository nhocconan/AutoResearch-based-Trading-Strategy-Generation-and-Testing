#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout + 1d ATR-based volatility filter + chop regime
# - Primary: 4h Donchian channel breakout (20-period) for directional bias
# - HTF: 1d ATR ratio (current ATR(7) / ATR(30)) for volatility regime (low ATR ratio = low vol breakout)
# - Chop filter: CHOP(14) < 50 = trending market (avoid false breakouts in ranging)
# - Long: Price breaks above Donchian upper + low volatility regime (ATR ratio < 0.8) + trending (CHOP < 50)
# - Short: Price breaks below Donchian lower + low volatility regime (ATR ratio < 0.8) + trending (CHOP < 50)
# - Exit: Opposite Donchian breakout OR volatility expansion (ATR ratio > 1.2) OR chop shifts to ranging (CHOP > 60)
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Works in bull/bear: Donchian captures breakouts, low volatility filter avoids choppy false signals, chop filter confirms trend
# - Target: 75-200 trades over 4 years (19-50/year) to stay within fee drag limits for 4h timeframe

name = "4h_1d_donchian_atr_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough data for ATR and chop calculations
        return np.zeros(n)
    
    # Pre-compute 4h data
    close_4h = prices['close'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    volume_4h = prices['volume'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 4h Donchian Channel (20-period)
    lookback_donch = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(lookback_donch - 1, n):
        if not np.isnan(high_4h[i-lookback_donch+1:i+1]).any() and not np.isnan(low_4h[i-lookback_donch+1:i+1]).any():
            upper[i] = np.max(high_4h[i-lookback_donch+1:i+1])
            lower[i] = np.min(low_4h[i-lookback_donch+1:i+1])
    
    # Calculate 4h Donchian breakout signals
    breakout_up = np.zeros(n, dtype=bool)
    breakout_down = np.zeros(n, dtype=bool)
    for i in range(lookback_donch, n):
        if not np.isnan(close_4h[i]) and not np.isnan(upper[i-1]) and not np.isnan(lower[i-1]):
            breakout_up[i] = close_4h[i] > upper[i-1]
            breakout_down[i] = close_4h[i] < lower[i-1]
    
    # Calculate 1d ATR(7) and ATR(30) for volatility regime filter
    atr_period_short = 7
    atr_period_long = 30
    
    # True Range for 1-period
    tr1 = np.maximum(np.maximum(high_1d - low_1d,
                               np.abs(np.roll(high_1d, 1) - low_1d)),
                    np.abs(np.roll(low_1d, 1) - high_1d))
    
    # ATR(7)
    atr7 = np.full(len(tr1), np.nan)
    for i in range(atr_period_short - 1, len(tr1)):
        if not np.isnan(tr1[i-atr_period_short+1:i+1]).any():
            atr7[i] = np.mean(tr1[i-atr_period_short+1:i+1])
    
    # ATR(30)
    atr30 = np.full(len(tr1), np.nan)
    for i in range(atr_period_long - 1, len(tr1)):
        if not np.isnan(tr1[i-atr_period_long+1:i+1]).any():
            atr30[i] = np.mean(tr1[i-atr_period_long+1:i+1])
    
    # ATR ratio: ATR(7) / ATR(30)
    atr_ratio = np.full(len(tr1), np.nan)
    for i in range(len(tr1)):
        if not np.isnan(atr7[i]) and not np.isnan(atr30[i]) and atr30[i] > 0:
            atr_ratio[i] = atr7[i] / atr30[i]
        else:
            atr_ratio[i] = np.nan
    
    # Calculate 1d Chopiness Index (CHOP) for regime filter
    chop_lookback = 14
    
    # True Range for 1-period (already calculated as tr1)
    
    # Sum of TR over chop_lookback period
    sum_tr = np.full(len(tr1), np.nan)
    for i in range(chop_lookback, len(tr1)):
        if not np.isnan(tr1[i-chop_lookback:i]).any():
            sum_tr[i] = np.sum(tr1[i-chop_lookback:i])
    
    # Highest high and lowest low over chop_lookback period
    hh = np.full(len(high_1d), np.nan)
    ll = np.full(len(low_1d), np.nan)
    for i in range(chop_lookback, len(high_1d)):
        if not np.isnan(high_1d[i-chop_lookback:i+1]).any() and not np.isnan(low_1d[i-chop_lookback:i+1]).any():
            hh[i] = np.max(high_1d[i-chop_lookback:i+1])
            ll[i] = np.min(low_1d[i-chop_lookback:i+1])
    
    # Chopiness Index
    chop = np.full(len(high_1d), np.nan)
    for i in range(chop_lookback, len(high_1d)):
        if (not np.isnan(sum_tr[i]) and not np.isnan(hh[i]) and not np.isnan(ll[i]) and 
            hh[i] > ll[i] and sum_tr[i] > 0):
            chop[i] = 100 * np.log10(sum_tr[i] / (hh[i] - ll[i])) / np.log10(chop_lookback)
        else:
            chop[i] = np.nan
    
    # Align all HTF indicators to 4h timeframe
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(lookback_donch, n):  # Start after Donchian warmup period
        # Skip if any required data is invalid
        if (np.isnan(atr_ratio_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility confirmation: low ATR ratio (< 0.8) indicates low volatility environment
        vol_confirm = atr_ratio_aligned[i] < 0.8
        
        # Chop regime filter: CHOP < 50 indicates trending market (avoid ranging)
        regime_confirm = chop_aligned[i] < 50.0
        
        # Donchian breakout signals
        donchian_up = breakout_up[i]
        donchian_down = breakout_down[i]
        
        # Exit conditions: 
        # 1. Opposite Donchian breakout
        # 2. Volatility expansion (ATR ratio > 1.2) 
        # 3. Chop shifts to ranging (CHOP > 60)
        exit_long = donchian_down or (atr_ratio_aligned[i] > 1.2) or (chop_aligned[i] > 60.0)
        exit_short = donchian_up or (atr_ratio_aligned[i] > 1.2) or (chop_aligned[i] > 60.0)
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Donchian breakout up + low volatility regime + trending
            if donchian_up and vol_confirm and regime_confirm:
                position = 1
                signals[i] = 0.25
            # Short entry: Donchian breakout down + low volatility regime + trending
            elif donchian_down and vol_confirm and regime_confirm:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Opposite Donchian breakout OR volatility expansion OR chop shifts to ranging
            if position == 1:  # Long position
                if exit_long:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if exit_short:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals