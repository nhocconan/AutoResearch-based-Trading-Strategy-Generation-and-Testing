#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout + 1d volume spike + 1w choppiness regime filter
# - Long: price breaks above Donchian(20) upper band + 1d volume > 1.5x 20-period volume average + 1w Choppiness Index > 61.8 (ranging market)
# - Short: price breaks below Donchian(20) lower band + 1d volume > 1.5x 20-period volume average + 1w Choppiness Index > 61.8 (ranging market)
# - Exit: price reverses back to Donchian midpoint or opposite band touch
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Target: 20-50 trades/year to stay within fee drag limits
# - Choppiness filter ensures we only trade in ranging markets where mean reversion works
# - Works in both bull and bear markets by focusing on ranging conditions where price respects Donchian channels

name = "4h_1d_1w_donchian_chop_v1"
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
    
    # Load 4h data ONCE before loop for Donchian calculation (MTF rule compliance)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return signals
    
    # Load 1d data ONCE before loop for volume confirmation (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Load 1w data ONCE before loop for Choppiness Index (MTF rule compliance)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return signals
    
    # Pre-compute 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Donchian upper band = highest high over 20 periods
    donchian_upper = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    # Donchian lower band = lowest low over 20 periods
    donchian_lower = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    # Donchian midpoint = (upper + lower) / 2
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    # Align Donchian levels to 4h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid)
    
    # Pre-compute 1d volume SMA for confirmation
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Pre-compute 1w Choppiness Index (14-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1_w = pd.Series(high_1w).rolling(2).max() - pd.Series(low_1w).rolling(2).min()
    tr2_w = abs(pd.Series(high_1w).shift(1) - pd.Series(close_1w))
    tr3_w = abs(pd.Series(low_1w).shift(1) - pd.Series(close_1w))
    tr_w = pd.concat([tr1_w, tr2_w, tr3_w], axis=1).max(axis=1)
    atr_w = tr_w.rolling(window=14, min_periods=14).mean().values
    
    # Choppiness Index = 100 * log10(sum(ATR14) / (max(high) - min(low))) / log10(14)
    atr_sum_14 = pd.Series(atr_w).rolling(window=14, min_periods=14).sum().values
    hhvl_1w = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    llvl_1w = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    chop_numerator = atr_sum_14 / (hhvl_1w - llvl_1w + 1e-10)
    chop_numerator = np.where(chop_numerator > 0, chop_numerator, 1e-10)
    chop = 100 * np.log10(chop_numerator) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or
            np.isnan(donchian_mid_aligned[i]) or np.isnan(volume_sma_20_1d_aligned[i]) or
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        high_price = high[i]
        low_price = low[i]
        
        # Volume confirmation: 1d volume > 1.5x 20-period volume average (tighter threshold)
        volume_1d_current = df_1d['volume'].values
        volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d_current)
        vol_confirm = volume_1d_aligned[i] > 1.5 * volume_sma_20_1d_aligned[i]
        
        # Weekly chop filter: Choppiness Index > 61.8 (ranging market)
        weekly_chop = chop_aligned[i]
        chop_filter = weekly_chop > 61.8
        
        # Donchian breakout conditions
        donchian_breakout_long = close_price > donchian_upper_aligned[i]
        donchian_breakout_short = close_price < donchian_lower_aligned[i]
        
        # Entry conditions
        enter_long = donchian_breakout_long and vol_confirm and chop_filter
        enter_short = donchian_breakout_short and vol_confirm and chop_filter
        
        # Exit conditions: price reverses to midpoint or touches opposite band
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if price drops below midpoint or touches lower band (mean reversion)
            exit_long = close_price < donchian_mid_aligned[i] or low_price <= donchian_lower_aligned[i]
        elif position == -1:
            # Exit short if price rises above midpoint or touches upper band (mean reversion)
            exit_short = close_price > donchian_mid_aligned[i] or high_price >= donchian_upper_aligned[i]
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals