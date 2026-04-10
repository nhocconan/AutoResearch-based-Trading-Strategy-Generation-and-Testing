#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume spike and ADX trend filter
# - Enter long when 4h price breaks above Donchian upper(20) AND 1d volume > 1.5x 20-period volume SMA AND 1d ADX(14) > 25
# - Enter short when 4h price breaks below Donchian lower(20) AND 1d volume > 1.5x 20-period volume SMA AND 1d ADX(14) > 25
# - Exit: price returns to Donchian middle (10-period average of upper/lower) or opposite band touch
# - Donchian breakout captures volatility expansion
# - Volume confirmation ensures breakouts have participation
# - ADX filter ensures we only trade in trending markets (avoids chop)
# - Target: 20-40 trades/year to minimize fee drag while capturing high-probability breakouts

name = "4h_1d_donchian_volume_adx_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for volume and ADX filters (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Pre-compute Donchian channels for 4h data (20-period)
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = highest_high_20
    donchian_lower = lowest_low_20
    donchian_middle = (donchian_upper + donchian_lower) / 2  # Middle channel
    
    # Pre-compute volume SMA for 1d data (20-period)
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Pre-compute ADX for 1d data (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d).diff().abs()
    tr2 = (pd.Series(high_1d) - pd.Series(close_1d.shift(1))).abs()
    tr3 = (pd.Series(low_1d) - pd.Series(close_1d.shift(1))).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    up_move = pd.Series(high_1d).diff()
    down_move = -pd.Series(low_1d).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM and TR
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    tr_smooth = pd.Series(atr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_1d = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Pre-compute 1d volume aligned
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    for i in range(20, n):  # Start after 20-bar warmup for Donchian
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(donchian_middle[i]) or 
            np.isnan(volume_sma_20_1d_aligned[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 1d volume > 1.5x 20-period volume SMA
        vol_confirm = volume_1d_aligned[i] > 1.5 * volume_sma_20_1d_aligned[i]
        
        # Trend filter: 1d ADX > 25 (trending market)
        trend_filter = adx_1d_aligned[i] > 25
        
        # Donchian breakout signals
        breakout_up = close[i] > donchian_upper[i] and close[i-1] <= donchian_upper[i-1]  # Cross above upper band
        breakout_down = close[i] < donchian_lower[i] and close[i-1] >= donchian_lower[i-1]  # Cross below lower band
        
        # Exit conditions
        exit_long = close[i] < donchian_middle[i]  # Return to middle band
        exit_short = close[i] > donchian_middle[i]  # Return to middle band
        exit_opposite_long = close[i] < donchian_lower[i]  # Touch lower band while long
        exit_opposite_short = close[i] > donchian_upper[i]  # Touch upper band while short
        
        # Trading logic
        if vol_confirm and trend_filter:
            # Long: Donchian breakout above upper band
            if breakout_up:
                if position != 1:  # Only signal on new long entry
                    position = 1
                    signals[i] = 0.25
                else:
                    signals[i] = 0.25
            # Short: Donchian breakout below lower band
            elif breakout_down:
                if position != -1:  # Only signal on new short entry
                    position = -1
                    signals[i] = -0.25
                else:
                    signals[i] = -0.25
            else:
                # Check for exits
                if position == 1 and (exit_long or exit_opposite_long):
                    position = 0
                    signals[i] = 0.0
                elif position == -1 and (exit_short or exit_opposite_short):
                    position = 0
                    signals[i] = 0.0
                else:
                    # Maintain current position
                    signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
        else:
            # No volume confirmation or not trending: exit any position
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals