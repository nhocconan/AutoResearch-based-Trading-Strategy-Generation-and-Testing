#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 12h ADX regime filter and volume confirmation.
# Long when Bull Power > 0, ADX > 25 (trending), and volume > 1.5x 20-period average.
# Short when Bear Power < 0, ADX > 25 (trending), and volume > 1.5x 20-period average.
# Exit when Elder Ray power reverses sign or ADX < 20 (range market).
# Uses discrete position size 0.25. Designed to capture trend strength with volume confirmation.
# Works in both bull and bear markets by requiring ADX > 25 to ensure we only trade strong trends,
# avoiding whipsaws in ranging conditions. Elder Ray provides dynamic bull/bear strength measurement.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Indicators: EMA13 for Elder Ray ===
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # === 12h Indicators: ADX(14) for trend strength ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = pd.Series(high_12h).diff()
    tr2 = pd.Series(low_12h).diff().abs()
    tr3 = pd.Series(close_12h).shift(1).diff().abs()
    tr_12h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_12h = pd.Series(tr_12h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    up_move = pd.Series(high_12h).diff()
    down_move = pd.Series(low_12h).diff().abs()
    up_move = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    down_move = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    
    # Smoothed DM
    plus_dm_12h = pd.Series(up_move).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_12h = pd.Series(down_move).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI and ADX
    plus_di_12h = 100 * plus_dm_12h / atr_12h
    minus_di_12h = 100 * minus_dm_12h / atr_12h
    dx_12h = 100 * np.abs(plus_di_12h - minus_di_12h) / (plus_di_12h + minus_di_12h)
    adx_12h = pd.Series(dx_12h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # === 12h Indicators: EMA20 for Elder Ray reference ===
    ema_20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    # === 12h Indicators: Volume MA for confirmation ===
    vol_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(vol_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    volume_spike = volume > (1.5 * vol_ma_12h_aligned)
    
    # === Elder Ray Components ===
    bull_power = high - ema_20_12h_aligned  # Bull Power = High - EMA(20)
    bear_power = low - ema_20_12h_aligned   # Bear Power = Low - EMA(20)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed)
    warmup = 60
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(ema_13[i]) or np.isnan(adx_12h_aligned[i]) or np.isnan(ema_20_12h_aligned[i]) or
            np.isnan(volume_spike[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        bp = bull_power[i]
        br = bear_power[i]
        adx_val = adx_12h_aligned[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if Bull Power becomes negative OR ADX drops below 20 (range)
            if bp <= 0 or adx_val < 20:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if Bear Power becomes positive OR ADX drops below 20 (range)
            if br >= 0 or adx_val < 20:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Bull Power > 0, ADX > 25 (strong trend), volume spike
            if bp > 0 and adx_val > 25 and vol_spike:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Bear Power < 0, ADX > 25 (strong trend), volume spike
            elif br < 0 and adx_val > 25 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_ElderRay_12hADX_VolumeSpike_V1"
timeframe = "6h"
leverage = 1.0