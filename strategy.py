#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d volume spike and 1w ADX trend filter
# - Enter long when price breaks above Donchian(20) upper band AND 1d volume > 2.0x 20-period volume SMA AND 1w ADX > 25
# - Enter short when price breaks below Donchian(20) lower band AND 1d volume > 2.0x 20-period volume SMA AND 1w ADX > 25
# - Exit: price moves to Donchian(20) opposite band or ATR-based trailing stop
# - Donchian provides clear breakout levels with built-in volatility adaptation
# - Volume confirmation ensures institutional participation and reduces false breakouts
# - 1w ADX filter ensures we only trade in trending markets (avoiding chop)
# - Target: 12-30 trades/year to minimize fee drag while capturing strong trends

name = "6h_1d_1w_donchian_voladx_v1"
timeframe = "6h"
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
    
    # Load 1d data ONCE before loop for volume confirmation (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return signals
    
    # Load 1w data ONCE before loop for ADX trend filter (MTF rule compliance)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return signals
    
    # Pre-compute Donchian channels for 6h data (20-period)
    # Upper band = highest high over 20 periods
    # Lower band = lowest low over 20 periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Pre-compute ATR for 6h data (14-period) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute volume SMA for 1d data (20-period)
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Pre-compute ADX for 1w data (14-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate True Range for 1w
    tr1_1w = high_1w - low_1w
    tr2_1w = np.abs(high_1w - np.roll(close_1w, 1))
    tr3_1w = np.abs(low_1w - np.roll(close_1w, 1))
    tr1_1w[0] = 0
    tr2_1w[0] = 0
    tr3_1w[0] = 0
    tr_1w = np.maximum(tr1_1w, np.maximum(tr2_1w, tr3_1w))
    
    # Calculate +DM and -DM
    up_move = high_1w - np.roll(high_1w, 1)
    down_move = np.roll(low_1w, 1) - low_1w
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    plus_dm_14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values
    minus_dm_14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values
    
    # Calculate DI+ and DI-
    plus_di_14 = 100 * plus_dm_14 / tr_14
    minus_di_14 = 100 * minus_dm_14 / tr_14
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align 1w ADX to 6h timeframe (wait for completed 1w bar)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Align volume SMA to 6h timeframe
    volume_1d_current = df_1d['volume'].values
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d_current)
    volume_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Pre-compute 1w close aligned for reference (not used in logic but available)
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
    
    for i in range(20, n):  # Start after 20-bar warmup for Donchian
        # Skip if any required data is invalid
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(atr[i]) or
            np.isnan(volume_sma_20_1d_aligned[i]) or np.isnan(volume_1d_aligned[i]) or
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 1d volume > 2.0x 20-period volume SMA
        vol_confirm = volume_1d_aligned[i] > 2.0 * volume_sma_20_1d_aligned[i]
        
        # Trend filter: 1w ADX > 25 (trending market)
        trending = adx_aligned[i] > 25
        
        # Breakout signals
        breakout_up = close[i] > donchian_upper[i-1]  # Close above previous upper band
        breakout_down = close[i] < donchian_lower[i-1]  # Close below previous lower band
        
        # Exit conditions
        exit_long = close[i] < donchian_lower[i]  # Price closes below lower band
        exit_short = close[i] > donchian_upper[i]  # Price closes above upper band
        atr_stop_long = close[i] < (donchian_upper[i] - 2.0 * atr[i])  # 2x ATR stop from entry approximation
        atr_stop_short = close[i] > (donchian_lower[i] + 2.0 * atr[i])  # 2x ATR stop from entry approximation
        
        # Trading logic
        if vol_confirm and trending:
            # Long: upward breakout in trending market
            if breakout_up:
                if position != 1:  # Only signal on new long entry
                    position = 1
                    signals[i] = 0.25
                else:
                    signals[i] = 0.25
            # Short: downward breakout in trending market
            elif breakout_down:
                if position != -1:  # Only signal on new short entry
                    position = -1
                    signals[i] = -0.25
                else:
                    signals[i] = -0.25
            else:
                # Check for exits
                if position == 1 and (exit_long or atr_stop_long):
                    position = 0
                    signals[i] = 0.0
                elif position == -1 and (exit_short or atr_stop_short):
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