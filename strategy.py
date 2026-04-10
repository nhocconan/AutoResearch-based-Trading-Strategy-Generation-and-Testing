#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with 1d volume spike confirmation and chop regime filter
# - Long when price breaks above 4h Donchian upper (20) AND 1d volume > 1.5x 20-period volume SMA AND chop > 61.8 (ranging market)
# - Short when price breaks below 4h Donchian lower (20) AND 1d volume > 1.5x 20-period volume SMA AND chop > 61.8
# - Exit: price retraces to midpoint of Donchian channel OR volume drops below average
# - Position sizing: 0.30 discrete level to balance risk and reward
# - Target: 50-100 trades/year on 4h timeframe to stay within fee drag limits
# - Uses Donchian structure for breakouts, volume spike for confirmation, chop filter to avoid whipsaws in trends

name = "4h_1d_donchian_volume_chop_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 4h Donchian channel (20-period)
    donchian_len = 20
    donchian_upper = pd.Series(high).rolling(window=donchian_len, min_periods=donchian_len).max().values
    donchian_lower = pd.Series(low).rolling(window=donchian_len, min_periods=donchian_len).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # Calculate 1d volume SMA for confirmation
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Calculate 4h volume SMA for confirmation
    volume_sma_20_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Chopiness Index (14-period) on 1d for regime filter
    # Chop = 100 * log10(sum(ATR(14)) / (log10(n) * (HHV(high,14) - LLV(low,14))))
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # ATR(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of ATR(14) over 14 periods
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    
    # HHV and LLV of 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chopiness Index
    chop_denom = np.log10(14) * (hh_14 - ll_14)
    chop_1d = np.where(chop_denom != 0, 100 * np.log10(sum_atr_14 / chop_denom), 50.0)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    for i in range(50, n):  # Start after warmup for indicators
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(volume_sma_20_4h[i]) or np.isnan(volume_sma_20_1d_aligned[i]) or
            np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: both 4h and 1d volume > 1.5x their 20-period SMA
        vol_confirm_4h = volume[i] > 1.5 * volume_sma_20_4h[i]
        vol_confirm_1d = volume_1d[i // 24] > 1.5 * volume_sma_20_1d_aligned[i] if i // 24 < len(volume_1d) else False
        vol_confirm = vol_confirm_4h and vol_confirm_1d
        
        # Regime filter: chop > 61.8 indicates ranging market (good for mean reversion/breakouts)
        chop_filter = chop_1d_aligned[i] > 61.8
        
        # Donchian breakout signals (using previous bar's levels to avoid look-ahead)
        breakout_up = close[i] > donchian_upper[i-1]
        breakout_down = close[i] < donchian_lower[i-1]
        
        # Exit conditions: price retreats to midpoint OR loss of volume confirmation OR chop drops below 38.2 (trending)
        exit_long = close[i] < donchian_mid[i] or not vol_confirm or chop_1d_aligned[i] < 38.2
        exit_short = close[i] > donchian_mid[i] or not vol_confirm or chop_1d_aligned[i] < 38.2
        
        if position == 0:  # Flat - look for entry
            if breakout_up and vol_confirm and chop_filter:
                position = 1
                signals[i] = 0.30
            elif breakout_down and vol_confirm and chop_filter:
                position = -1
                signals[i] = -0.30
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
        else:  # position == -1 (Short position) - look for exit
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
    
    return signals