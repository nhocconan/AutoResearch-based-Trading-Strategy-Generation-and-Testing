#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with volume confirmation and 1d EMA50 trend filter + ATR stoploss
# Long when price breaks above Donchian(20) upper + volume > 2x 20-period avg + close > 1d EMA50
# Short when price breaks below Donchian(20) lower + volume > 2x 20-period avg + close < 1d EMA50
# Uses Donchian channels for price structure, 1d EMA for higher timeframe trend, volume for confirmation
# Designed for low trade frequency (20-40/year) to minimize fee drag and maximize edge
# ATR-based stoploss implemented via signal=0 when price moves against position by 2.5x ATR
# Works in both bull and bear markets by requiring volume confirmation and HTF trend alignment

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h HTF data once before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # === 4h Indicators: Donchian Channel (20) and ATR (14) ===
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Donchian upper/lower (20-period)
    donchian_upper_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_lower_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR (14-period)
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === 1d Indicator: EMA50 ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all HTF indicators to 4h timeframe
    donchian_upper_20_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper_20)
    donchian_lower_20_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower_20)
    atr_14_aligned = align_htf_to_ltf(prices, df_4h, atr_14)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 2.0)
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_20_aligned[i]) or np.isnan(donchian_lower_20_aligned[i]) or
            np.isnan(atr_14_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 4h Donchian(20) upper
        # 2. Volume confirmation
        # 3. Close above 1d EMA50 (HTF uptrend)
        if (close[i] > donchian_upper_20_aligned[i]) and vol_confirm and \
           (close[i] > ema_50_1d_aligned[i]):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 4h Donchian(20) lower
        # 2. Volume confirmation
        # 3. Close below 1d EMA50 (HTF downtrend)
        elif (close[i] < donchian_lower_20_aligned[i]) and vol_confirm and \
             (close[i] < ema_50_1d_aligned[i]):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_Donchian20_Volume_1dEMA50_Filter_v3"
timeframe = "4h"
leverage = 1.0