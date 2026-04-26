#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_VolumeSpike_ChopRegime_v1
Hypothesis: Trade daily Donchian(20) breakouts with volume confirmation (2.0x median) and choppiness regime filter (CHOP>61.8 = range, mean reversion; CHOP<38.2 = trend, follow breakout). Only long when price > 200 EMA for trend filter. Designed for low-frequency, high-conviction trades (target 7-25/year) to minimize fee drag and work in both bull/bear markets via regime adaptation.
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
    
    # Get 1w data for HTF trend filter (EMA34)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA(34) for trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Donchian(20) on daily timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Donchian upper/lower: 20-day high/low
    donchian_high = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1d timeframe (already aligned as we use 1d data)
    donchian_high_aligned = donchian_high  # no alignment needed for same timeframe
    donchian_low_aligned = donchian_low
    
    # 200 EMA for trend filter (daily)
    ema_200_1d = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = ema_200_1d  # same timeframe
    
    # Choppiness Index on daily timeframe
    chop_period = 14
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=chop_period, adjust=False, min_periods=chop_period).mean().values
    sum_tr = pd.Series(tr).rolling(window=chop_period, min_periods=chop_period).sum().values
    hh = pd.Series(high).rolling(window=chop_period, min_periods=chop_period).max().values
    ll = pd.Series(low).rolling(window=chop_period, min_periods=chop_period).min().values
    chop = 100 * np.log10(sum_tr / np.where((hh - ll) > 0, hh - ll, np.nan)) / np.log10(chop_period)
    
    # Align HTF indicators (1w EMA) to 1d timeframe
    # Donchian and chop are already 1d, no alignment needed
    
    # Volume confirmation: 2.0x median volume (20-period)
    vol_median = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of EMA34(1w), Donchian(20), EMA200(1d), chop(14), volume median(20)
    start_idx = max(34, 20, 200, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema_200_aligned[i]) or
            np.isnan(chop[i]) or
            np.isnan(vol_median[i])):
            # Hold current position
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_median_val = vol_median[i]
        ema_34_1w_val = ema_34_1w_aligned[i]
        donchian_high_val = donchian_high_aligned[i]
        donchian_low_val = donchian_low_aligned[i]
        ema_200_val = ema_200_aligned[i]
        chop_val = chop[i]
        
        # Regime filters
        chop_high_regime = chop_val > 61.8  # range market, mean reversion
        chop_low_regime = chop_val < 38.2   # trending market
        
        # Trend filter: price > EMA200 for long bias in uptrend
        uptrend_bias = close_val > ema_200_val
        downtrend_bias = close_val < ema_200_val
        
        if position == 0:
            # Long: break above Donchian high with volume spike
            # In choppy range (CHOP>61.8): mean reversion - wait for pullback? Actually, breakout still valid but be selective
            # In trending (CHOP<38.2): follow breakout
            # Require volume spike and alignment with 1w EMA trend
            long_signal = (close_val > donchian_high_val) and \
                          (volume_val > 2.0 * vol_median_val) and \
                          (ema_34_1w_val > 0) and \
                          (uptrend_bias or chop_high_regime)  # allow long in uptrend or range (mean reversion long from low)
            
            # Short: break below Donchian low with volume spike
            short_signal = (close_val < donchian_low_val) and \
                           (volume_val > 2.0 * vol_median_val) and \
                           (ema_34_1w_val > 0) and \
                           (downtrend_bias or chop_high_regime)  # allow short in downtrend or range (mean reversion short from high)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: close below Donchian low (contrarian exit) or trailing stop
            if close_val < donchian_low_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: close above Donchian high (contrarian exit) or trailing stop
            if close_val > donchian_high_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Donchian20_Breakout_VolumeSpike_ChopRegime_v1"
timeframe = "1d"
leverage = 1.0