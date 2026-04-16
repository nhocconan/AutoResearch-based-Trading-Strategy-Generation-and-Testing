#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot R1/S1 breakout with 1d volume confirmation and 1h session filter.
# Long when price breaks above 4h Camarilla R1 AND 1d volume > 1.2x 20-period average AND hour in 08-20 UTC.
# Short when price breaks below 4h Camarilla S1 AND 1d volume > 1.2x 20-period average AND hour in 08-20 UTC.
# Exit when price retests the 4h pivot point (PP) or ATR-based stoploss (1.5*ATR from entry).
# Uses discrete position size 0.20. Designed to capture intraday breakouts with volume confirmation in active sessions.
# Target: 80-180 total trades over 4 years (20-45/year) to balance opportunity and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h Indicators: Camarilla Pivot Points (based on prior 4h bar) ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Prior 4h bar's OHLC for Camarilla calculation
    prior_high_4h = np.roll(high_4h, 1)
    prior_low_4h = np.roll(low_4h, 1)
    prior_close_4h = np.roll(close_4h, 1)
    # Set first value to NaN (no prior bar)
    prior_high_4h[0] = np.nan
    prior_low_4h[0] = np.nan
    prior_close_4h[0] = np.nan
    
    # Camarilla formulas
    pp_4h = (prior_high_4h + prior_low_4h + prior_close_4h) / 3
    range_4h = prior_high_4h - prior_low_4h
    r1_4h = pp_4h + (range_4h * 1.1 / 12)
    s1_4h = pp_4h - (range_4h * 1.1 / 12)
    
    # Align to 1h timeframe (wait for completed 4h bar)
    pp_4h_aligned = align_htf_to_ltf(prices, df_4h, pp_4h)
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    
    # === 1d Indicators: Volume Spike (volume > 1.2x 20-period average) ===
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (1.2 * vol_ma_1d_aligned)
    
    # Session filter: 08-20 UTC (using DatetimeIndex)
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 20 periods needed for volume MA)
    warmup = 50
    
    # Track position state and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Calculate 4h ATR for stoploss
    tr1_4h = pd.Series(high_4h).diff()
    tr2_4h = pd.Series(low_4h).diff().abs()
    tr3_4h = pd.Series(close_4h).shift(1).diff().abs()
    tr_4h = pd.concat([tr1_4h, tr2_4h, tr3_4h], axis=1).max(axis=1)
    atr_4h_raw = pd.Series(tr_4h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h_raw)
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(pp_4h_aligned[i]) or np.isnan(r1_4h_aligned[i]) or np.isnan(s1_4h_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(atr_4h_aligned[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        atr_val = atr_4h_aligned[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price retests pivot point (PP)
            if price <= pp_4h_aligned[i]:
                exit_signal = True
            # ATR-based stoploss: 1.5*ATR below entry
            elif price < entry_price - 1.5 * atr_val:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price retests pivot point (PP)
            if price >= pp_4h_aligned[i]:
                exit_signal = True
            # ATR-based stoploss: 1.5*ATR above entry
            elif price > entry_price + 1.5 * atr_val:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Camarilla R1 AND volume spike AND in session
            if price > r1_4h_aligned[i] and vol_spike:
                signals[i] = 0.20
                position = 1
                entry_price = price
            
            # SHORT: Price breaks below Camarilla S1 AND volume spike AND in session
            elif price < s1_4h_aligned[i] and vol_spike:
                signals[i] = -0.20
                position = -1
                entry_price = price
        
        else:
            signals[i] = position * 0.20
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dVolumeSpike_1hSession_V1"
timeframe = "1h"
leverage = 1.0