#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot R4/S4 breakout with 12h volume spike confirmation and ATR-based position sizing
# Long when price > Camarilla R4 AND 12h volume > 2.5x 20-period volume SMA
# Short when price < Camarilla S4 AND 12h volume > 2.5x 20-period volume SMA
# Exit on price returning to Camarilla pivot point (PP) or ATR stoploss (2.0 ATR)
# Uses discrete position sizing (0.25) to limit fee drag and Camarilla levels provide objective structure proven in ranging/ bear markets
# Volume filter reduces false breakouts; pivot points work across regimes
# R4/S4 levels are more extreme than R3/S3, reducing trade frequency and improving edge

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
    
    # === 4h Indicator: Camarilla Pivot Levels (based on previous day) ===
    # Camarilla levels calculated from prior day's OHLC
    # PP = (H + L + C) / 3
    # R4 = PP + (H - L) * 1.1
    # S4 = PP - (H - L) * 1.1
    # We use 1d data to calculate daily pivots, then align to 4h
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    pp_1d = (h_1d + l_1d + c_1d) / 3.0
    r4_1d = pp_1d + (h_1d - l_1d) * 1.1
    s4_1d = pp_1d - (h_1d - l_1d) * 1.1
    
    # Align 1d levels to 4h timeframe (wait for completed 1d bar)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # ATR for stoploss (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(30, 20)  # 12h volume SMA and 1d data need ~30 bars
    
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
        if (np.isnan(pp_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(vol_sma_20_12h_aligned[i]) or 
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # Current 12h volume (aligned)
        vol_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_12h)
        if np.isnan(vol_12h_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 12h volume > 2.5x 20-period 12h volume SMA
        vol_threshold = vol_sma_20_12h_aligned[i] * 2.5
        vol_confirm = vol_12h_aligned[i] > vol_threshold
        
        # Price levels
        price = close[i]
        pp = pp_aligned[i]
        r4 = r4_aligned[i]
        s4 = s4_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        if position == 1:  # long position
            # Exit on price returning to pivot point or ATR stoploss
            if price <= pp or price <= entry_price - 2.0 * atr_14[i]:
                exit_signal = True
        elif position == -1:  # short position
            # Exit on price returning to pivot point or ATR stoploss
            if price >= pp or price >= entry_price + 2.0 * atr_14[i]:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG CONDITIONS
            # Price > Camarilla R4 AND volume confirmation
            if price > r4 and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT CONDITIONS
            # Price < Camarilla S4 AND volume confirmation
            elif price < s4 and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = 0.0  # maintain position
    
    return signals

name = "4h_Camarilla_R4S4_12hVolSpike2.5x_v1"
timeframe = "4h"
leverage = 1.0