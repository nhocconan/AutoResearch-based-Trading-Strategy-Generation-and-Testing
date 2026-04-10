#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ATR volume spike filter and 1d ADX trend regime
# - Long when price breaks above 20-period 4h Donchian high + volume > 1.5x 20-period 1d ATR-scaled volume + ADX(14) > 25
# - Short when price breaks below 20-period 4h Donchian low + volume > 1.5x 20-period 1d ATR-scaled volume + ADX(14) > 25
# - Exit: price returns to 20-period 4h Donchian midpoint (mean reversion to equilibrium)
# - Position sizing: 0.25 discrete level
# - Donchian channels identify volatility-based support/resistance
# - ATR-scaled volume confirmation ensures breakouts have conviction
# - ADX > 25 ensures trending environment to avoid false breakouts in ranging markets
# - Works in bull/bear: breakouts in both directions, ADX filter ensures trending conditions

name = "4h_1d_donchian_atr_volume_adx_v3"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate 4h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Calculate 1d ATR (14-period) for volume scaling
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = high_1d[0] - low_1d[0]  # first bar TR
    
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d ATR-scaled volume (volume / ATR) and its 20-period SMA
    volume_1d = df_1d['volume'].values
    volume_atr_ratio_1d = volume_1d / atr_1d
    volume_atr_sma_20_1d = pd.Series(volume_atr_ratio_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d ADX (14-period) for trend regime filter
    # +DM and -DM calculation
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed +DM, -DM, and TR
    tr_atr = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    # +DI and -DI
    plus_di = 100 * plus_dm_smooth / tr_atr
    minus_di = 100 * minus_dm_smooth / tr_atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_1d = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align all HTF indicators to 4h timeframe (completed 1d bar only)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid)
    volume_atr_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_atr_sma_20_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(volume_atr_sma_20_1d_aligned[i]) or 
            np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d ATR-scaled volume > 1.5x its 20-period SMA
        vol_atr_ratio_current = align_htf_to_ltf(prices, df_1d, volume_atr_ratio_1d)
        vol_confirm = vol_atr_ratio_current[i] > 1.5 * volume_atr_sma_20_1d_aligned[i]
        
        # Regime filter: ADX > 25 indicates trending market
        regime_filter = adx_1d_aligned[i] > 25
        
        # Donchian breakout entry conditions
        # Long: price breaks above Donchian high + volume confirmation + trending regime
        # Short: price breaks below Donchian low + volume confirmation + trending regime
        long_entry = (close[i] > donchian_high_aligned[i] and 
                     vol_confirm and 
                     regime_filter)
        short_entry = (close[i] < donchian_low_aligned[i] and 
                      vol_confirm and 
                      regime_filter)
        
        # Exit conditions: price returns to Donchian midpoint (mean reversion)
        exit_long = close[i] < donchian_mid_aligned[i]
        exit_short = close[i] > donchian_mid_aligned[i]
        
        if position == 0:  # Flat - look for entry
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals