#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot S1/R1 breakout with 1w EMA50 trend filter and volume confirmation
# Long when price breaks above R1 AND 1w EMA50 uptrend AND 1d volume > 1.5x 20-median
# Short when price breaks below S1 AND 1w EMA50 downtrend AND 1d volume > 1.5x 20-median
# Exit when price reverts to Camarilla pivot point (PP) or ATR stop (2.0)
# Position size 0.25. Target: 40-80 trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 80:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1w data once before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 55:
        return np.zeros(n)
    
    # === 1w Indicators ===
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Get 1d data once before loop for Camarilla pivots and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Indicators ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Camarilla pivot levels (based on previous day's OHLC)
    # PP = (H + L + C) / 3
    # S1 = C - (H - L) * 1.1 / 12
    # S2 = C - (H - L) * 1.1 / 6
    # S3 = C - (H - L) * 1.1 / 4
    # R1 = C + (H - L) * 1.1 / 12
    # R2 = C + (H - L) * 1.1 / 6
    # R3 = C + (H - L) * 1.1 / 4
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = close_1d + (high_1d - low_1d) * 1.1 / 12.0
    s1_1d = close_1d - (high_1d - low_1d) * 1.1 / 12.0
    
    # Align Camarilla levels to 1d timeframe (no shift needed as they're based on completed daily bar)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # 1d volume median (20-period) for spike detection
    vol_median_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).median().values
    vol_median_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_median_20_1d)
    
    # ATR for stoploss (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(55, 30, 20, 14)  # 1w EMA50, 1d Camarilla, volume median, ATR
    
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
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(pp_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(vol_median_20_1d_aligned[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # Current 1d volume (aligned)
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
        if np.isnan(vol_1d_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 1d volume > 1.5x 20-period 1d volume median
        vol_threshold = vol_median_20_1d_aligned[i] * 1.5
        vol_confirm = vol_1d_aligned[i] > vol_threshold
        
        # Trend filter: 1w EMA50 direction
        trend_up = close[i] > ema50_1w_aligned[i]
        trend_down = close[i] < ema50_1w_aligned[i]
        
        price = close[i]
        pp_level = pp_aligned[i]
        r1_level = r1_aligned[i]
        s1_level = s1_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        if position == 1:  # long position
            # Exit on price reversion to pivot point or ATR stoploss
            if price <= pp_level or price <= entry_price - 2.0 * atr_14[i]:
                exit_signal = True
        elif position == -1:  # short position
            # Exit on price reversion to pivot point or ATR stoploss
            if price >= pp_level or price >= entry_price + 2.0 * atr_14[i]:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG CONDITIONS
            # Price breaks above R1 AND uptrend AND volume confirmation
            if price > r1_level and trend_up and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
            
            # SHORT CONDITIONS
            # Price breaks below S1 AND downtrend AND volume confirmation
            elif price < s1_level and trend_down and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        else:
            signals[i] = 0.0  # maintain position
    
    return signals

name = "1d_Camarilla_R1S1_1wEMA50_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0