#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h volume confirmation and ATR-based position sizing
# Long when price > Donchian upper band AND 12h volume > 1.8x 20-period volume SMA
# Short when price < Donchian lower band AND 12h volume > 1.8x 20-period volume SMA
# Exit on price returning to Donchian middle band or ATR stoploss (2.0 ATR)
# Uses discrete position sizing (0.25) to limit fee drag
# Donchian channels provide objective structure that works in trending markets
# Volume filter reduces false breakouts; works across regimes
# Target: 75-200 total trades over 4 years (19-50/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 12h data once before loop for volume filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # === 12h Indicator: Volume SMA (20-period) for confirmation ===
    volume_12h = df_12h['volume'].values
    vol_sma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_sma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_sma_20_12h)
    
    # === 4h Indicator: Donchian Channels (20-period) ===
    # Upper band = highest high over 20 periods
    # Lower band = lowest low over 20 periods
    # Middle band = (upper + lower) / 2
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2.0
    
    # ATR for stoploss (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(30, 20)  # 12h volume SMA and Donchian need ~30 bars
    
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
            np.isnan(donchian_middle[i]) or np.isnan(vol_sma_20_12h_aligned[i]) or 
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # Current 12h volume (aligned)
        vol_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_12h)
        if np.isnan(vol_12h_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 12h volume > 1.8x 20-period 12h volume SMA
        vol_threshold = vol_sma_20_12h_aligned[i] * 1.8
        vol_confirm = vol_12h_aligned[i] > vol_threshold
        
        # Price levels
        price = close[i]
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        middle = donchian_middle[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        if position == 1:  # long position
            # Exit on price returning to middle band or ATR stoploss
            if price <= middle or price <= entry_price - 2.0 * atr_14[i]:
                exit_signal = True
        elif position == -1:  # short position
            # Exit on price returning to middle band or ATR stoploss
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
            # Price > Donchian upper band AND volume confirmation
            if price > upper and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT CONDITIONS
            # Price < Donchian lower band AND volume confirmation
            elif price < lower and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = 0.0  # maintain position
    
    return signals

name = "4h_Donchian20_12hVolSpike1.8x_v1"
timeframe = "4h"
leverage = 1.0