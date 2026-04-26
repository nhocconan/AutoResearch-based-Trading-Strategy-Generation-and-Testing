#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike
Hypothesis: Camarilla R1/S1 breakouts with 4h trend filter and volume confirmation capture strong momentum moves.
In bull markets: price breaks above R1 (first resistance) with 4h uptrend and volume spike → long.
In bear markets: price breaks below S1 (first support) with 4h downtrend and volume spike → short.
Uses discrete sizing (0.30) and ATR-based stoploss to limit drawdown.
Target: 50-150 trades over 4 years. Camarilla pivots from 1d provide institutional levels that work across regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:  # Need warmup for ATR and volume
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation: volume > 2.0x 20-period median
    vol_series = pd.Series(volume)
    vol_median = vol_series.rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (vol_median * 2.0)
    
    # ATR for stoploss (14-period)
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Load daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day's OHLC
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    typical_price = (h_1d + l_1d + c_1d) / 3.0
    hl_range = h_1d - l_1d
    
    r1_1d = typical_price + (hl_range * 1.1 / 12.0)
    s1_1d = typical_price - (hl_range * 1.1 / 12.0)
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # 4h EMA50 for trend filter
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.30
    bars_since_entry = 0
    
    # Start after warmup (need 50 for EMA)
    start_idx = 50
    
    for i in range(start_idx, n):
        bars_since_entry += 1
        
        # Skip if any data not ready
        if (np.isnan(ema_50[i]) or 
            np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or 
            np.isnan(volume_spike[i]) or
            np.isnan(atr[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        close_val = close[i]
        ema_val = ema_50[i]
        r1_val = r1_1d_aligned[i]
        s1_val = s1_1d_aligned[i]
        atr_val = atr[i]
        
        # Long logic: price breaks above R1 with volume spike and 4h uptrend
        long_condition = (close_val > r1_val) and volume_spike[i] and (close_val > ema_val)
        # Short logic: price breaks below S1 with volume spike and 4h downtrend
        short_condition = (close_val < s1_val) and volume_spike[i] and (close_val < ema_val)
        
        # Stoploss logic: ATR-based
        stop_long = position == 1 and close_val < (entry_price - 2.0 * atr_val)
        stop_short = position == -1 and close_val > (entry_price + 2.0 * atr_val)
        
        # Minimum holding period: 2 bars
        if position != 0 and bars_since_entry < 2:
            # Hold position regardless of signals
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
            bars_since_entry = 0
            entry_price = close_val
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
            bars_since_entry = 0
            entry_price = close_val
        elif position == 1 and (exit_long := close_val < ema_val or stop_long):
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
        elif position == -1 and (exit_short := close_val > ema_val or stop_short):
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0