#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian breakout with 1d volume confirmation and 1w trend filter.
Uses Donchian channel breakouts (20-period) for entry, filtered by 1d volume > 1.5x 20-day average
and 1w EMA50 trend direction. Exits when price crosses the opposite Donchian band or volume drops.
Designed to capture strong trending moves with volume confirmation while avoiding false breakouts.
Target: 20-40 trades/year by requiring confluence of breakout, volume, and trend filters.
Works in bull markets (long on breakouts above upper band) and bear markets (short on breakdowns below lower band).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Donchian Channel (20-period) on 4h ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate Donchian bands
    donch_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align to 4h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_4h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_4h, donch_low)
    
    # === 1d Volume Confirmation ===
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # === 1w EMA50 Trend Filter ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    
    # Warmup: need enough data for Donchian and EMA calculations
    warmup = 60
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or
            np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-day average
        vol_today_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        vol_confirm = vol_today_aligned[i] > vol_ma_20_1d_aligned[i] * 1.5
        
        # Trend filter: price above/below weekly EMA50
        price_vs_ema = close[i] > ema_50_1w_aligned[i]  # True for uptrend
        
        # Breakout conditions
        breakout_up = high[i] > donch_high_aligned[i]  # Price breaks above upper band
        breakdown_down = low[i] < donch_low_aligned[i]  # Price breaks below lower band
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: bullish breakout with volume confirmation and uptrend
            if breakout_up and vol_confirm and price_vs_ema:
                signals[i] = 0.25
                position = 1
                continue
            # Short: bearish breakdown with volume confirmation and downtrend
            elif breakdown_down and vol_confirm and not price_vs_ema:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price breaks below lower Donchian band or volume fails
            if low[i] < donch_low_aligned[i] or not vol_confirm:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above upper Donchian band or volume fails
            if high[i] > donch_high_aligned[i] or not vol_confirm:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dVolumeConfirm_1wEMA50Trend"
timeframe = "4h"
leverage = 1.0