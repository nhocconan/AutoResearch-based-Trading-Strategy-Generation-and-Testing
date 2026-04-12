#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Williams %R extreme + 1d volume spike + choppiness regime filter
    # Williams %R < -80 = oversold (long), > -20 = overbought (short) on 12h
    # 1d volume > 2.0x 20-period MA confirms institutional participation
    # 1d choppiness > 61.8 = ranging market (mean reversion), < 38.2 = trending (avoid)
    # Only trade mean reversion in ranging regimes to avoid whipsaws in trends
    # Target: 12-37 trades/year per symbol (50-150 total over 4 years)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Session filter: 8:00-20:00 UTC (avoid low volume Asian session)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 12h data for Williams %R
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 12h Williams %R(14)
    highest_high_12h = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    lowest_low_12h = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    williams_r_12h = -100 * (highest_high_12h - close_12h) / (highest_high_12h - lowest_low_12h + 1e-10)
    williams_r_12h_aligned = align_htf_to_ltf(prices, df_12h, williams_r_12h)
    
    # Get 1d data for volume and choppiness
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # 1d volume spike: current volume > 2.0x 20-period MA
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (2.0 * vol_ma_1d)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    # 1d choppiness index: CHOP(14) = 100 * log10(SUM(TR(14)) / (ATR(14) * 14)) / log10(14)
    # CHOP > 61.8 = ranging (good for mean reversion), CHOP < 38.2 = trending (avoid)
    tr_1d = np.maximum(
        high_1d - low_1d,
        np.maximum(
            np.abs(high_1d - np.roll(close_1d, 1)),
            np.abs(low_1d - np.roll(close_1d, 1))
        )
    )
    tr_1d[0] = high_1d[0] - low_1d[0]  # first TR
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    sum_tr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    chop_denominator = atr_1d * 14
    chop_denominator = np.where(chop_denominator == 0, 1e-10, chop_denominator)
    chop_1d = 100 * np.log10(sum_tr_14 / chop_denominator) / np.log10(14)
    chop_1d = np.where(np.isnan(chop_1d), 50.0, chop_1d)  # neutral if undefined
    chop_regime_1d = chop_1d > 61.8  # True = ranging (good for mean reversion)
    chop_regime_aligned = align_htf_to_ltf(prices, df_1d, chop_regime_1d.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if data not ready
        if (np.isnan(williams_r_12h_aligned[i]) or 
            np.isnan(volume_spike_aligned[i]) or 
            np.isnan(chop_regime_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Williams %R extremes: < -80 oversold (long), > -20 overbought (short)
        williams_r = williams_r_12h_aligned[i]
        oversold = williams_r < -80
        overbought = williams_r > -20
        
        # Only trade in ranging markets (chop > 61.8) with volume confirmation
        in_ranging = chop_regime_aligned[i] > 0.5
        vol_spike = volume_spike_aligned[i] > 0.5
        
        # Mean reversion entry: fade extremes in ranging markets with volume
        long_entry = oversold and in_ranging and vol_spike
        short_entry = overbought and in_ranging and vol_spike
        
        # Exit when Williams %R returns to neutral range (-80 to -20)
        long_exit = williams_r > -50  # exit on recovery from oversold
        short_exit = williams_r < -50  # exit on decline from overbought
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_williamsr_vol_chop_v2"
timeframe = "12h"
leverage = 1.0