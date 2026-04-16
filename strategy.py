#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h volume spike (>2.0x median) and 12h EMA50 trend filter
# Uses tighter volume threshold (2.0x) and EMA50 for trend to reduce trades and work in bull/bear markets
# Long when price > upper Donchian(20) AND 12h volume > 2.0x 20-period 12h volume median AND close > 12h EMA50
# Short when price < lower Donchian(20) AND 12h volume > 2.0x 20-period 12h volume median AND close < 12h EMA50
# Exit on price returning to Donchian midpoint or ATR stoploss (2.0 ATR)
# Position size 0.25 to limit fee drag. Target: 75-200 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 12h data once before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # === 12h Indicators ===
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    volume_12h = df_12h['volume'].values
    
    # 12h EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # 12h volume median (20-period) for spike detection
    vol_median_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).median().values
    vol_median_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_median_20_12h)
    
    # ATR for stoploss (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(60, 20, 50)  # 12h EMA50, volume median, Donchian channels
    
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
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(vol_median_20_12h_aligned[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # Current 12h volume (aligned)
        vol_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_12h)
        if np.isnan(vol_12h_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 12h volume > 2.0x 20-period 12h volume median (tighter threshold)
        vol_threshold = vol_median_20_12h_aligned[i] * 2.0
        vol_confirm = vol_12h_aligned[i] > vol_threshold
        
        # Trend filter: price vs 12h EMA50
        price = close[i]
        trend_long = price > ema50_12h_aligned[i]
        trend_short = price < ema50_12h_aligned[i]
        
        # Price levels
        upper = donchian_high[i]
        lower = donchian_low[i]
        midpoint = donchian_mid[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        if position == 1:  # long position
            # Exit on price returning to midpoint or ATR stoploss
            if price <= midpoint or price <= entry_price - 2.0 * atr_14[i]:
                exit_signal = True
        elif position == -1:  # short position
            # Exit on price returning to midpoint or ATR stoploss
            if price >= midpoint or price >= entry_price + 2.0 * atr_14[i]:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG CONDITIONS
            # Price > upper Donchian AND volume confirmation AND uptrend (price > EMA50)
            if price > upper and vol_confirm and trend_long:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT CONDITIONS
            # Price < lower Donchian AND volume confirmation AND downtrend (price < EMA50)
            elif price < lower and vol_confirm and trend_short:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = 0.0  # maintain position
    
    return signals

name = "4h_Donchian20_12hVolMedian2.0x_EMA50_v1"
timeframe = "4h"
leverage = 1.0