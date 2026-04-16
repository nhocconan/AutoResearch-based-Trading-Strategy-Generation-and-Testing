#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R1/S1 breakout with 12h volume confirmation and ATR volatility filter.
# Long when close breaks above R1 with volume > 1.2x 12h average volume and ATR(14) > 0.5 * ATR(50).
# Short when close breaks below S1 with same volume and volatility filters.
# Exit on opposite Camarilla level (S1 for longs, R1 for shorts) or ATR trailing stop (3*ATR from extreme).
# Uses discrete position size 0.25. Camarilla levels from 12h pivot provide intraday structure.
# Volume and volatility filters avoid false breakouts in low-momentum environments.
# Works in bull/bear by requiring both volume expansion and volatility expansion for breakout validity.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h HTF: Camarilla pivot levels (R1, S1) and ATR filters ===
    df_12h = get_htf_data(prices, '12h')
    h_12h = df_12h['high'].values
    l_12h = df_12h['low'].values
    c_12h = df_12h['close'].values
    
    # Calculate pivot point (PP) = (H + L + C) / 3
    pp_12h = (h_12h + l_12h + c_12h) / 3.0
    # Calculate range
    range_12h = h_12h - l_12h
    # Camarilla levels
    r1_12h = pp_12h + (range_12h * 1.0 / 12.0)  # R1 = PP + (H-L) * 1/12
    s1_12h = pp_12h - (range_12h * 1.0 / 12.0)  # S1 = PP - (H-L) * 1/12
    
    # Align 12h levels to 6h
    r1_12h_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    s1_12h_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    
    # === 12h Volume and ATR filters ===
    vol_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    volume_spike = volume > (1.2 * vol_ma_12h_aligned)
    
    # ATR(14) and ATR(50) for volatility expansion filter
    tr1_12h = pd.Series(h_12h).diff()
    tr2_12h = pd.Series(l_12h).diff().abs()
    tr3_12h = pd.Series(c_12h).shift(1).diff().abs()
    tr_12h = pd.concat([tr1_12h, tr2_12h, tr3_12h], axis=1).max(axis=1)
    atr_14_12h = pd.Series(tr_12h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_50_12h = pd.Series(tr_12h).ewm(alpha=1/50, adjust=False, min_periods=50).mean().values
    atr_14_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_14_12h)
    atr_50_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_50_12h)
    volatility_expansion = atr_14_12h_aligned > (0.5 * atr_50_12h_aligned)
    
    # === 6h ATR for trailing stop ===
    tr1_6h = pd.Series(high).diff()
    tr2_6h = pd.Series(low).diff().abs()
    tr3_6h = pd.Series(close).shift(1).diff().abs()
    tr_6h = pd.concat([tr1_6h, tr2_6h, tr3_6h], axis=1).max(axis=1)
    atr_6h_raw = pd.Series(tr_6h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed)
    warmup = 100
    
    # Track position state and extremes for trailing stop
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    long_extreme = 0.0
    short_extreme = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(r1_12h_aligned[i]) or np.isnan(s1_12h_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(volatility_expansion[i]) or
            np.isnan(atr_6h_raw[i]) or not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        vol_exp = volatility_expansion[i]
        atr_val = atr_6h_raw[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price breaks below S1 (failed breakout)
            if price < s1_12h_aligned[i]:
                exit_signal = True
            # ATR trailing stop: 3*ATR below long extreme
            elif price < long_extreme - 3.0 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price breaks above R1 (failed breakdown)
            if price > r1_12h_aligned[i]:
                exit_signal = True
            # ATR trailing stop: 3*ATR above short extreme
            elif price > short_extreme + 3.0 * atr_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            long_extreme = 0.0
            short_extreme = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Close breaks above R1 with volume spike and volatility expansion
            if (close[i] > r1_12h_aligned[i] and close[i-1] <= r1_12h_aligned[i-1] and
                vol_spike and vol_exp):
                signals[i] = 0.25
                position = 1
                entry_price = price
                long_extreme = price
            
            # SHORT: Close breaks below S1 with volume spike and volatility expansion
            elif (close[i] < s1_12h_aligned[i] and close[i-1] >= s1_12h_aligned[i-1] and
                  vol_spike and vol_exp):
                signals[i] = -0.25
                position = -1
                entry_price = price
                short_extreme = price
        
        else:
            # Update extremes for trailing stop
            if position == 1:
                long_extreme = max(long_extreme, price)
                signals[i] = 0.25
            elif position == -1:
                short_extreme = min(short_extreme, price)
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R1_S1_Breakout_Volume_Volatility_V1"
timeframe = "6h"
leverage = 1.0