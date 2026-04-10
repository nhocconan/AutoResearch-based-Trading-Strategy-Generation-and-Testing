#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d ADX trend filter and volume confirmation
# - Long: price breaks above Camarilla H3 level + 1d ADX > 25 + volume > 1.8x 20-period average
# - Short: price breaks below Camarilla L3 level + 1d ADX > 25 + volume > 1.8x 20-period average
# - Uses discrete position sizing (0.25) to minimize fee churn
# - ATR-based stoploss (2.5x ATR(14)) to manage risk
# - Designed for 4h timeframe: targets 25-40 trades/year to avoid fee drag
# - Works in bull/bear markets: ADX filter ensures strong trend, Camarilla levels provide structure

name = "4h_1d_camarilla_pivot_breakout_adx_volume_v1"
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
    
    # Pre-compute 1d ADX(14) for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed DM and TR
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values
    tr_smooth = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    
    # Calculate DI and DX
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_14 = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Pre-compute 4h Camarilla pivot levels (based on previous day)
    # Camarilla levels: H4 = close + 1.5*(high-low), H3 = close + 1.25*(high-low), etc.
    # We need previous day's OHLC for today's levels
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    
    # Calculate Camarilla levels for 1d
    camarilla_h3_1d = prev_close + 1.1 * (prev_high - prev_low)  # H3 = close + 1.1*(high-low)
    camarilla_l3_1d = prev_close - 1.1 * (prev_high - prev_low)  # L3 = close - 1.1*(high-low)
    camarilla_h3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3_1d)
    camarilla_l3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3_1d)
    
    # Pre-compute 4h Donchian channels (20-period) for exit
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 4h volume confirmation
    volume_4h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_4h > (1.8 * avg_volume_20)
    
    # Pre-compute 4h ATR(14) for stoploss
    tr1_4h = high_4h - low_4h
    tr2_4h = np.abs(high_4h - np.roll(close_4h, 1))
    tr3_4h = np.abs(low_4h - np.roll(close_4h, 1))
    tr_4h = np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))
    tr_4h[0] = tr1_4h[0]
    atr_14 = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(adx_14_aligned[i]) or np.isnan(vol_spike[i]) or
            np.isnan(atr_14[i]) or np.isnan(camarilla_h3_1d_aligned[i]) or
            np.isnan(camarilla_l3_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low OR stoploss hit
            if low_4h[i] < donchian_low[i] or close_4h[i] < entry_price - 2.5 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high OR stoploss hit
            if high_4h[i] > donchian_high[i] or close_4h[i] > entry_price + 2.5 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla breakout with ADX and volume filters
            if vol_spike[i] and adx_14_aligned[i] > 25:
                # Long: price breaks above Camarilla H3 level
                if high_4h[i] > camarilla_h3_1d_aligned[i]:
                    position = 1
                    entry_price = close_4h[i]
                    signals[i] = 0.25
                # Short: price breaks below Camarilla L3 level
                elif low_4h[i] < camarilla_l3_1d_aligned[i]:
                    position = -1
                    entry_price = close_4h[i]
                    signals[i] = -0.25
    
    return signals