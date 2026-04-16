#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 12h ADX regime filter and volume confirmation.
# Bull Power = High - EMA(13); Bear Power = EMA(13) - Low.
# Long when Bull Power > 0 AND rising AND ADX > 25 (trending) AND volume > 1.5x 20-period average.
# Short when Bear Power > 0 AND rising AND ADX > 25 AND volume > 1.5x 20-period average.
# Exit when power turns negative or opposite signal triggers.
# Uses discrete position size 0.25. Elder Ray measures trend strength via price relative to EMA.
# ADX filter ensures we only trade in trending markets, avoiding whipsaws in ranges.
# Volume confirmation adds conviction to breakouts.
# Target: 50-150 total trades over 4 years (12-37/year) to balance edge and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Indicators: EMA(13) for Elder Ray ===
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13  # Higher highs relative to trend = bullish strength
    bear_power = ema_13 - low   # Lower lows relative to trend = bearish strength
    # Rising power: current > previous
    bull_power_rising = bull_power > np.roll(bull_power, 1)
    bear_power_rising = bear_power > np.roll(bear_power, 1)
    
    # === 12h Indicators: ADX(14) for regime filter ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = pd.Series(high_12h).diff()
    tr2 = pd.Series(low_12h).diff().abs()
    tr3 = pd.Series(close_12h).shift(1).diff().abs()
    tr_12h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_12h = pd.Series(tr_12h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    up_move = pd.Series(high_12h).diff()
    down_move = pd.Series(low_12h).diff().abs()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM and TR
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    tr_smooth = pd.Series(tr_12h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx_12h = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    adx_strong = adx_12h_aligned > 25  # Trending market
    
    # === 6h Volume Spike (volume > 1.5x 20-period average) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed)
    warmup = 50
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(bull_power_rising[i]) or
            np.isnan(bear_power_rising[i]) or np.isnan(adx_12h_aligned[i]) or np.isnan(volume_spike[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        adx_ok = adx_strong[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if Bull Power turns negative (losing bullish momentum)
            if bull_power[i] <= 0:
                exit_signal = True
            # Optional: exit on opposite Bear Power signal
            elif bear_power[i] > 0 and bear_power_rising[i]:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if Bear Power turns negative (losing bearish momentum)
            if bear_power[i] <= 0:
                exit_signal = True
            # Optional: exit on opposite Bull Power signal
            elif bull_power[i] > 0 and bull_power_rising[i]:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Bull Power > 0 AND rising AND ADX > 25 AND volume spike
            if bull_power[i] > 0 and bull_power_rising[i] and adx_ok and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Bear Power > 0 AND rising AND ADX > 25 AND volume spike
            elif bear_power[i] > 0 and bear_power_rising[i] and adx_ok and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_ElderRay_12hADX_VolumeSpike_V1"
timeframe = "6h"
leverage = 1.0