#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d volume spike + chop regime filter
# - Long when price breaks above 4h Donchian upper channel + 1d volume > 1.5x 20-period average + chop < 61.8 (trending)
# - Short when price breaks below 4h Donchian lower channel + 1d volume > 1.5x 20-period average + chop < 61.8 (trending)
# - Exit when price reverts to 4h Donchian midpoint or volume drops below average
# - Uses ATR-based stoploss (signal→0 when adverse move > 2*ATR)
# - Target: 20-40 trades/year to minimize fee drag while capturing strong trending moves
# - Works in bull/bear: Donchian breakouts capture momentum; volume confirms conviction; chop filter avoids whipsaws in ranging markets

name = "4h_1d_donchian_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    # Load 4h data ONCE before loop for Donchian channels and ATR (MTF rule compliance)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return signals
    
    # Load 1d data ONCE before loop for volume and chop (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Pre-compute 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Align Donchian channels to 4h timeframe (already in 4h, but align for consistency)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid)
    
    # Pre-compute 4h ATR(14) for stoploss
    tr1 = pd.Series(high_4h).rolling(2).max() - pd.Series(low_4h).rolling(2).min()
    tr2 = abs(pd.Series(high_4h).shift(1) - pd.Series(close_4h := df_4h['close'].values))
    tr3 = abs(pd.Series(low_4h).shift(1) - pd.Series(close_4h))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_4h, atr_14)
    
    # Pre-compute 1d volume SMA for confirmation
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Pre-compute 1d chop regime filter (Choppiness Index)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1_1d = pd.Series(high_1d).rolling(2).max() - pd.Series(low_1d).rolling(2).min()
    tr2_1d = abs(pd.Series(high_1d).shift(1) - pd.Series(close_1d))
    tr3_1d = abs(pd.Series(low_1d).shift(1) - pd.Series(close_1d))
    tr_1d = pd.concat([tr1_1d, tr2_1d, tr3_1d], axis=1).max(axis=1)
    
    # Sum of TR over 14 periods
    sum_tr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(sum_tr_14 / (hh_14 - ll_14)) / log10(14)
    # Avoid division by zero and log of zero/negative
    denominator = hh_14 - ll_14
    chop_raw = np.where((denominator > 0) & (sum_tr_14 > 0), 
                        100 * np.log10(sum_tr_14 / denominator) / np.log10(14), 
                        50.0)  # Default to neutral when invalid
    chop = chop_raw
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(donchian_mid_aligned[i]) or np.isnan(atr_14_aligned[i]) or
            np.isnan(volume_sma_20_1d_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current 4h bar data
        close_4h_current = close[i]
        high_4h_current = high[i]
        low_4h_current = low[i]
        
        # Volume confirmation: 1d volume > 1.5x 20-period volume SMA
        volume_1d_current = df_1d['volume'].values
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d_current)
        vol_confirm = volume_1d_aligned[i] > 1.5 * volume_sma_20_1d_aligned[i]
        
        # Chop regime filter: chop < 61.8 indicates trending market (good for breakouts)
        chop_current = chop_aligned[i]
        trending_regime = chop_current < 61.8
        
        # Donchian breakout conditions
        breakout_up = close_4h_current > donchian_high_aligned[i]
        breakout_down = close_4h_current < donchian_low_aligned[i]
        
        # Entry conditions
        enter_long = breakout_up and vol_confirm and trending_regime
        enter_short = breakout_down and vol_confirm and trending_regime
        
        # Exit conditions
        exit_long = (position == 1 and 
                    (close_4h_current < donchian_mid_aligned[i] or  # Revert to midpoint
                     not vol_confirm or  # Volume drops
                     close_4h_current < entry_price - 2.0 * atr_14_aligned[i]))  # ATR stoploss
        exit_short = (position == -1 and 
                     (close_4h_current > donchian_mid_aligned[i] or  # Revert to midpoint
                      not vol_confirm or  # Volume drops
                      close_4h_current > entry_price + 2.0 * atr_14_aligned[i]))  # ATR stoploss
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            entry_price = close_4h_current
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            entry_price = close_4h_current
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            entry_price = 0.0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            entry_price = 0.0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals