#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 12h ADX regime filter and volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 12h ADX(14) to filter ranging (ADX < 20) vs trending (ADX > 25) markets.
- In trending regime (ADX > 25): Donchian breakout continuation (long on upper break, short on lower break).
- In ranging regime (ADX < 20): Donchian mean reversion (fade at bands - long at lower, short at upper).
- Volume confirmation: current volume > 1.5 * 6h volume MA(20) to ensure participation.
- Signal size: 0.25 discrete to balance capture and fee control.
- Designed for BTC/ETH: Donchian provides objective trend channels, ADX avoids whipsaws in ranging markets,
  volume filter ensures breakouts/mean reversions have conviction. Works in bull (trend continuation) 
  and bear (mean reversion in ranges, trend continuation in strong moves).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for Donchian calculation and volume
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    # Get 12h data for ADX regime filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:  # ADX needs ~30 bars for stability
        return np.zeros(n)
    
    # Calculate 12h ADX(14) for regime filter
    # TR calculation
    tr1 = df_6h['high'] - df_6h['low']
    tr2 = np.abs(df_6h['high'] - np.roll(df_6h['close'], 1))
    tr3 = np.abs(df_6h['low'] - np.roll(df_6h['close'], 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr_6h = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_6h = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    
    # +DM and -DM
    up_move = df_6h['high'] - np.roll(df_6h['high'], 1)
    down_move = np.roll(df_6h['low'], 1) - df_6h['low']
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed +DM, -DM, TR
    tr_period = 14
    if len(tr_6h) >= tr_period:
        atr_smooth = np.zeros_like(tr_6h)
        plus_dm_smooth = np.zeros_like(tr_6h)
        minus_dm_smooth = np.zeros_like(tr_6h)
        
        # Initial values
        atr_smooth[tr_period-1] = np.mean(tr_6h[:tr_period])
        plus_dm_smooth[tr_period-1] = np.mean(plus_dm[:tr_period])
        minus_dm_smooth[tr_period-1] = np.mean(minus_dm[:tr_period])
        
        # Wilder's smoothing
        for i in range(tr_period, len(tr_6h)):
            atr_smooth[i] = (atr_smooth[i-1] * (tr_period-1) + tr_6h[i]) / tr_period
            plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (tr_period-1) + plus_dm[i]) / tr_period
            minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (tr_period-1) + minus_dm[i]) / tr_period
        
        # DI and DX
        plus_di = 100 * plus_dm_smooth / atr_smooth
        minus_di = 100 * minus_dm_smooth / atr_smooth
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        
        # ADX (smoothed DX)
        adx_period = 14
        adx = np.full_like(dx, np.nan)
        if len(dx) >= adx_period:
            adx[adx_period-1] = np.mean(dx[:adx_period])
            for i in range(adx_period, len(dx)):
                adx[i] = (adx[i-1] * (adx_period-1) + dx[i]) / adx_period
    else:
        adx = np.full_like(tr_6h, np.nan)
    
    adx_12h_aligned = align_htf_to_ltf(prices, df_6h, adx)
    
    # Calculate 6h Donchian channels (20-period)
    donchian_upper = pd.Series(df_6h['high']).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(df_6h['low']).rolling(window=20, min_periods=20).min().values
    donchian_upper_aligned = align_htf_to_ltf(prices, df_6h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_6h, donchian_lower)
    
    # Calculate 6h volume MA(20) for confirmation
    volume_6h = df_6h['volume'].values
    vol_ma_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_ma_6h_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 20)  # ADX needs ~50, Donchian 20, volume MA 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_12h_aligned[i]) or np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or np.isnan(vol_ma_6h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Volume confirmation: 1.5x threshold (balanced to reduce noise)
        vol_confirm = curr_volume > 1.5 * vol_ma_6h_aligned[i]
        
        # Regime filters
        trending_regime = adx_12h_aligned[i] > 25
        ranging_regime = adx_12h_aligned[i] < 20
        
        if position == 0:
            # Check for entry signals
            if trending_regime and vol_confirm:
                # Trending: Donchian breakout continuation
                if curr_close > donchian_upper_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                elif curr_close < donchian_lower_aligned[i]:
                    signals[i] = -0.25
                    position = -1
            elif ranging_regime and vol_confirm:
                # Ranging: Donchian mean reversion (fade at bands)
                if curr_close < donchian_lower_aligned[i]:
                    signals[i] = 0.25  # Long at lower band (oversold)
                    position = 1
                elif curr_close > donchian_upper_aligned[i]:
                    signals[i] = -0.25  # Short at upper band (overbought)
                    position = -1
        elif position == 1:
            # Long position: exit on opposite signal or regime change
            if (trending_regime and curr_close < donchian_lower_aligned[i]) or \
               (ranging_regime and curr_close > donchian_upper_aligned[i]) or \
               (adx_12h_aligned[i] > 20 and adx_12h_aligned[i] < 25 and position == 1):  # Exit in transition
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit on opposite signal or regime change
            if (trending_regime and curr_close > donchian_upper_aligned[i]) or \
               (ranging_regime and curr_close < donchian_lower_aligned[i]) or \
               (adx_12h_aligned[i] > 20 and adx_12h_aligned[i] < 25 and position == -1):  # Exit in transition
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_12hADX_Regime_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0