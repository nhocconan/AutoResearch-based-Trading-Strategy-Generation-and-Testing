#!/usr/bin/env python3
"""
6h_KeltnerBreakout_VolumeRegime_1dTrend
Hypothesis: Keltner Channel breakouts with volume confirmation and 1d ADX regime filter capture strong momentum moves while avoiding choppy markets. 
In trending regimes (ADX>25): price breaks above/below 2.0*ATR Keltner bands with volume spike → follow breakout. 
In ranging regimes (ADX<20): fade breaks at bands as mean reversion. 
Uses 1d ADX for regime detection (more stable than lower timeframes) and volume confirmation to filter weak breakouts. 
Target: 50-150 trades over 4 years. Keltner channels adapt to volatility better than fixed Donchian bands.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:  # Need 30 for ATR and ADX
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation: volume > 1.5x 20-period median
    vol_series = pd.Series(volume)
    vol_median = vol_series.rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (vol_median * 1.5)
    
    # ATR for Keltner channels (20-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0]-low[0], np.abs(high[0]-close[0]), np.abs(low[0]-close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Keltner Channels: EMA(20) ± 2.0*ATR
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    upper_keltner = ema_20 + (2.0 * atr)
    lower_keltner = ema_20 - (2.0 * atr)
    
    # Load 1d data for HTF trend filter (ADX regime)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d ADX calculation (14-period)
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    
    # True Range
    tr1_1d = df_1d_high[1:] - df_1d_low[1:]
    tr2_1d = np.abs(df_1d_high[1:] - df_1d_close[:-1])
    tr3_1d = np.abs(df_1d_low[1:] - df_1d_close[:-1])
    tr_1d = np.concatenate([[np.max([df_1d_high[0]-df_1d_low[0], np.abs(df_1d_high[0]-df_1d_close[0]), np.abs(df_1d_low[0]-df_1d_close[0])])], np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))])
    
    # +DM and -DM
    up_move = df_1d_high[1:] - df_1d_high[:-1]
    down_move = df_1d_low[:-1] - df_1d_low[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing = EMA with alpha=1/period)
    atr_1d = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # +DI and -DI
    plus_di = 100 * (plus_dm_smooth / atr_1d)
    minus_di = 100 * (minus_dm_smooth / atr_1d)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 1d indicators to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    upper_keltner_aligned = align_htf_to_ltf(prices, df_1d, upper_keltner)  # Using 1d for HTF structure
    lower_keltner_aligned = align_htf_to_ltf(prices, df_1d, lower_keltner)
    ema_20_aligned = align_htf_to_ltf(prices, df_1d, ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    bars_since_entry = 0
    
    # Start after warmup (need 30 for ATR/ADX)
    start_idx = 30
    
    for i in range(start_idx, n):
        bars_since_entry += 1
        
        # Skip if any data not ready
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(upper_keltner_aligned[i]) or 
            np.isnan(lower_keltner_aligned[i]) or 
            np.isnan(ema_20_aligned[i]) or 
            np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        close_val = close[i]
        adx_val = adx_aligned[i]
        upper_val = upper_keltner_aligned[i]
        lower_val = lower_keltner_aligned[i]
        ema_val = ema_20_aligned[i]
        
        # Regime determination: ADX > 25 = trending, ADX < 20 = ranging
        is_trending = adx_val > 25
        is_ranging = adx_val < 20
        
        # Entry logic
        long_breakout = close_val > upper_val
        short_breakout = close_val < lower_val
        
        # Exit logic: opposite band touch or regime change to ranging
        exit_long = close_val < ema_val
        exit_short = close_val > ema_val
        
        # Minimum holding period: 2 bars
        if position != 0 and bars_since_entry < 2:
            # Hold position regardless of signals
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        if is_trending:
            # Trending regime: follow breakouts with volume confirmation
            if long_breakout and volume_spike[i] and position != 1:
                signals[i] = base_size
                position = 1
                bars_since_entry = 0
            elif short_breakout and volume_spike[i] and position != -1:
                signals[i] = -base_size
                position = -1
                bars_since_entry = 0
            elif position == 1 and exit_long:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            elif position == -1 and exit_short:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = base_size
                else:
                    signals[i] = -base_size
        elif is_ranging:
            # Ranging regime: mean reversion at bands
            if short_breakout and volume_spike[i] and position != -1:
                signals[i] = -base_size
                position = -1
                bars_since_entry = 0
            elif long_breakout and volume_spike[i] and position != 1:
                signals[i] = base_size
                position = 1
                bars_since_entry = 0
            elif position == 1 and exit_long:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            elif position == -1 and exit_short:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = base_size
                else:
                    signals[i] = -base_size
        else:
            # Transition regime (ADX 20-25): hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "6h_KeltnerBreakout_VolumeRegime_1dTrend"
timeframe = "6h"
leverage = 1.0