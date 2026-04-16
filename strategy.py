#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h ADX trend filter and volume confirmation
# Long when price > Donchian upper AND 12h ADX > 25 AND 12h volume > 1.5x 20-period volume SMA
# Short when price < Donchian lower AND 12h ADX > 25 AND 12h volume > 1.5x 20-period volume SMA
# Exit on Donchian middle line or ATR stoploss (2.0 ATR)
# Uses proven price channel structure, trend strength filter, volume confirmation
# Discrete sizing 0.25 targets 20-30 trades/year to minimize fee drag

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
    
    # Get 12h data once before loop for ADX and volume filters
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # === 12h Indicators: ADX(14) for trend strength ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ADX components
    plus_dm = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    minus_dm = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.concatenate([[close_12h[0]], close_12h[:-1]]))
    tr3 = np.abs(low_12h - np.concatenate([[close_12h[0]], close_12h[:-1]]))
    tr_12h = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_12h = pd.Series(tr_12h).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di_12h = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_12h
    minus_di_12h = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_12h
    dx_12h = 100 * np.abs(plus_di_12h - minus_di_12h) / (plus_di_12h + minus_di_12h)
    adx_12h = pd.Series(dx_12h).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # === 12h Indicator: Volume SMA (20-period) for confirmation ===
    volume_12h = df_12h['volume'].values
    vol_sma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_sma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_sma_20_12h)
    
    # === 4h Indicator: Donchian Channel (20-period) ===
    donchian_window = 20
    donchian_upper = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    donchian_lower = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2.0
    
    # ATR for stoploss (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(20, 30)  # Donchian(20) and 12h ADX needs ~30
    
    # Track position state for exits
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            if position != 0:
                position = 0  # force flat outside session
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_middle[i]) or np.isnan(adx_12h_aligned[i]) or 
            np.isnan(vol_sma_20_12h_aligned[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # Current 12h volume (aligned)
        vol_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_12h)
        if np.isnan(vol_12h_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 12h volume > 1.5x 20-period 12h volume SMA
        vol_threshold = vol_sma_20_12h_aligned[i] * 1.5
        vol_confirm = vol_12h_aligned[i] > vol_threshold
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_12h_aligned[i] > 25.0
        
        # Price levels
        price = close[i]
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        middle = donchian_middle[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        if position == 1:  # long position
            # Exit on price returning to middle line or ATR stoploss
            if price <= middle or price <= entry_price - 2.0 * atr_14[i]:
                exit_signal = True
        elif position == -1:  # short position
            # Exit on price returning to middle line or ATR stoploss
            if price >= middle or price >= entry_price + 2.0 * atr_14[i]:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG CONDITIONS
            # Price > Donchian upper AND strong trend AND volume confirmation
            if price > upper and strong_trend and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT CONDITIONS
            # Price < Donchian lower AND strong trend AND volume confirmation
            elif price < lower and strong_trend and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = 0.0  # maintain position
    
    return signals

name = "4h_Donchian20_12hADX_Volume1.5x_v1"
timeframe = "4h"
leverage = 1.0