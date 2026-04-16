#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1d volume confirmation and ATR filter.
# Uses 1d Camarilla levels (R3, S3, R4, S4) calculated from prior 1d OHLC.
# Long when 6h close breaks above R3 with volume > 1.5x 20-period 1d average and ATR(14) > 0.5*ATR(50) (volatility expansion).
# Short when 6h close breaks below S3 with same filters.
# Exit when price reverts to 1d pivot point (PP) or ATR-based stoploss (2*ATR from entry).
# Discrete position size 0.25. Designed to capture institutional breakout/continuation moves with volume and volatility confirmation.
# Works in both bull and bear markets by requiring volume spike and volatility expansion, avoiding false breakouts in low-volume ranging markets.
# Target: 50-150 total trades over 4 years (12-37/year) to balance edge and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1d Camarilla Pivot Levels (from prior 1d OHLC) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate prior 1d Camarilla levels (shifted by 1 to avoid look-ahead)
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    r4_1d = pp_1d + range_1d * 1.1 / 2
    r3_1d = pp_1d + range_1d * 1.1 / 4
    s3_1d = pp_1d - range_1d * 1.1 / 4
    s4_1d = pp_1d - range_1d * 1.1 / 2
    
    # Align to 6h timeframe (use prior 1d levels for current 6h bar)
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # === 1d Volume Confirmation (volume > 1.5x 20-period average) ===
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (1.5 * vol_ma_1d_aligned)
    
    # === 6h ATR Filter (ATR(14) > 0.5*ATR(50) for volatility expansion) ===
    tr1 = pd.Series(high).diff()
    tr2 = pd.Series(low).diff().abs()
    tr3 = pd.Series(close).shift(1).diff().abs()
    tr_6h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = pd.Series(tr_6h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_50 = pd.Series(tr_6h).ewm(alpha=1/50, adjust=False, min_periods=50).mean().values
    vol_expansion = atr_14 > (0.5 * atr_50)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 50 periods needed)
    warmup = 100
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(pp_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(vol_expansion[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        is_vol_exp = vol_expansion[i]
        pp = pp_1d_aligned[i]
        r3 = r3_1d_aligned[i]
        s3 = s3_1d_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price reverts to 1d pivot point
            if price <= pp:
                exit_signal = True
            # ATR-based stoploss: 2*ATR below entry
            elif 'atr_14' in locals() and price < entry_price - 2.0 * atr_14[i]:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price reverts to 1d pivot point
            if price >= pp:
                exit_signal = True
            # ATR-based stoploss: 2*ATR above entry
            elif 'atr_14' in locals() and price > entry_price + 2.0 * atr_14[i]:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Close breaks above R3 with volume spike and volatility expansion
            if price > r3 and vol_spike and is_vol_exp:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT: Close breaks below S3 with volume spike and volatility expansion
            elif price < s3 and vol_spike and is_vol_exp:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_Camarilla_R3_S3_Breakout_Volume_Volatility_V1"
timeframe = "6h"
leverage = 1.0