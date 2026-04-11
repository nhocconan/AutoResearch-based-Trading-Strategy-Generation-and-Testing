#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + ADX regime filter
# - Bull Power = High - EMA13, Bear Power = EMA13 - Low
# - Trend regime: ADX > 25 (trending), ADX < 20 (range)
# - In trending regime (ADX > 25): Long when Bull Power > 0 and rising, Short when Bear Power < 0 and falling
# - In range regime (ADX < 20): Mean reversion at Bollinger Bands (20,2.0)
# - Uses 1d EMA13 and ADX for HTF regime, aligned to 6h
# - Works in both bull and bear markets by adapting to regime
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits

name = "6h_1d_elder_ray_adx_regime_v1"
timeframe = "6h"
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
    
    # Load 1d data ONCE before loop for Elder Ray and ADX (MTF rule compliance)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return signals
    
    # Pre-compute 1d EMA13 for Elder Ray
    close_1d = df_1d['close'].values
    ema13 = pd.Series(close_1d).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Pre-compute 1d ADX
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first bar
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    # Smoothed DM
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values / atr
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Align 1d indicators to 6h timeframe
    ema13_aligned = align_htf_to_ltf(prices, df_1d, ema13)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    plus_di_aligned = align_htf_to_ltf(prices, df_1d, plus_di)
    minus_di_aligned = align_htf_to_ltf(prices, df_1d, minus_di)
    
    # Pre-compute 6h Bollinger Bands for range regime
    close_s = pd.Series(close)
    basis = close_s.rolling(window=20, min_periods=20).mean().values
    dev = close_s.rolling(window=20, min_periods=20).std().values
    upper_band = basis + 2.0 * dev
    lower_band = basis - 2.0 * dev
    
    # Pre-compute 6h volume confirmation
    volume = prices['volume'].values
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(ema13_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(plus_di_aligned[i]) or np.isnan(minus_di_aligned[i]) or
            np.isnan(basis[i]) or np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        close_price = close[i]
        high_price = high[i]
        low_price = low[i]
        volume_current = volume[i]
        
        # Elder Ray Power
        bull_power = high_price - ema13_aligned[i]
        bear_power = ema13_aligned[i] - low_price
        
        # Volume confirmation: current volume > 1.3x 20-period average
        vol_confirm = volume_current > 1.3 * volume_sma_20[i]
        
        # Regime filters
        adx_val = adx_aligned[i]
        is_trending = adx_val > 25
        is_range = adx_val < 20
        
        # Bollinger Band position for range regime
        bb_position = 0.5
        if upper_band[i] > lower_band[i]:
            bb_position = (close_price - lower_band[i]) / (upper_band[i] - lower_band[i])
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        if is_trending and vol_confirm:
            # Trending regime: follow Elder Ray momentum
            # Long: Bull Power > 0 and rising (vs previous bar)
            if i > 100:
                prev_bull_power = high[i-1] - ema13_aligned[i-1]
                enter_long = bull_power > 0 and bull_power > prev_bull_power
            else:
                enter_long = bull_power > 0
            
            # Short: Bear Power > 0 and rising (vs previous bar)
            if i > 100:
                prev_bear_power = ema13_aligned[i-1] - low[i-1]
                enter_short = bear_power > 0 and bear_power > prev_bear_power
            else:
                enter_short = bear_power > 0
        
        elif is_range and vol_confirm:
            # Range regime: mean reversion at Bollinger Bands
            enter_long = bb_position < 0.2 and close_price < basis[i]  # Near lower band
            enter_short = bb_position > 0.8 and close_price > basis[i]  # Near upper band
        
        # Exit conditions
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if trend weakens or reverses
            exit_long = (is_trending and bull_power <= 0) or (is_range and bb_position > 0.5)
        elif position == -1:
            # Exit short if trend weakens or reverses
            exit_short = (is_trending and bear_power <= 0) or (is_range and bb_position < 0.5)
        
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