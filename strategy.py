#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull Power/Bear Power) with 1d ADX regime filter and volume confirmation.
# Bull Power = High - EMA13, Bear Power = EMA13 - Low. Long when Bull Power > 0 AND Bear Power rising (less negative) AND ADX > 25 (trending) AND volume > 1.5x 20-period average.
# Short when Bear Power < 0 AND Bull Power falling (less positive) AND ADX > 25 AND volume > 1.5x 20-period average.
# Uses discrete position size 0.25. Designed to capture strong directional moves in trending markets while avoiding chop via ADX filter.
# Works in both bull and bear markets by requiring ADX > 25 for trend strength and volume confirmation to avoid false signals.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag while capturing high-probability trends.

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
    bull_power = high - ema_13  # Bull Power: High - EMA
    bear_power = ema_13 - low   # Bear Power: EMA - Low
    
    # === 1d Indicators: ADX(14) for trend strength ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = pd.Series(high_1d).diff()
    tr2 = pd.Series(low_1d).diff().abs()
    tr3 = pd.Series(close_1d).shift(1).diff().abs()
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = pd.Series(tr_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    up_move = pd.Series(high_1d).diff()
    down_move = pd.Series(low_1d).diff().abs()
    up_move = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    down_move = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    
    # Directional Indicators
    plus_di = 100 * (pd.Series(up_move).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_1d)
    minus_di = 100 * (pd.Series(down_move).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr_1d)
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    adx_trending = adx_aligned > 25
    
    # === 1d Indicators: Volume Spike (volume > 1.5x 20-period average) ===
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (1.5 * vol_ma_1d_aligned)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed for ADX/EMA)
    warmup = 100
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(volume_spike[i]) or not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        adx_val = adx_trending[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if Bear Power becomes positive (momentum fading)
            if bear_power[i] > 0:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if Bull Power becomes negative (momentum fading)
            if bull_power[i] < 0:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Bull Power > 0 AND Bear Power rising (less negative) AND ADX trending AND volume spike
            bear_power_rising = (i > warmup and bear_power[i] > bear_power[i-1])
            if bull_power[i] > 0 and bear_power_rising and adx_val and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Bear Power < 0 AND Bull Power falling (less positive) AND ADX trending AND volume spike
            elif bear_power[i] < 0:
                bull_power_falling = (i > warmup and bull_power[i] < bull_power[i-1])
                if bull_power_falling and adx_val and vol_spike:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_ElderRay_1dADX_VolumeSpike_V1"
timeframe = "6h"
leverage = 1.0