#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_WeeklyTrend_VolumeSpike
Hypothesis: On daily timeframe, Camarilla R1/S1 breakout in the direction of weekly trend (price > weekly EMA50 = bullish, price < weekly EMA50 = bearish) with volume confirmation (>1.5x 20-period MA) captures high-probability trend continuation moves. Weekly EMA50 acts as dynamic support/resistance and regime filter. Discrete position sizing (±0.25) and ATR-based trailing stop (2.0x) for exits. Targets 15-25 trades/year by requiring weekly regime alignment, volume confirmation, and Camarilla breakout structure—designed to work in both bull (breakouts above weekly EMA50) and bear (breakdowns below weekly EMA50) markets with BTC/ETH edge from institutional pivot levels and volume-confirmed breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for EMA50 trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    close_1w = pd.Series(df_1w['close'].values)
    weekly_ema50 = close_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_ema50_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema50)
    
    # Daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Camarilla pivot levels from previous day: based on (H+L+C)/3
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical price for pivot calculation
    typical_price = (high_1d + low_1d + close_1d) / 3.0
    # Camarilla width = (High - Low) * 1.1 / 12
    camarilla_width = (high_1d - low_1d) * 1.1 / 12.0
    # R1 = Close + width * 1.1, S1 = Close - width * 1.1
    camarilla_r1 = close_1d + camarilla_width * 1.1
    camarilla_s1 = close_1d - camarilla_width * 1.1
    
    # Align Camarilla levels to 1d timeframe (wait for completed 1d bar)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Daily ATR(14) for trailing stop
    tr1 = pd.Series(high).diff().abs()
    tr2 = (pd.Series(high) - pd.Series(close).shift()).abs()
    tr3 = (pd.Series(low) - pd.Series(close).shift()).abs()
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr_1d.ewm(span=14, adjust=False, min_periods=14).mean()
    atr_1d_values = atr_1d.values
    
    # Volume spike filter: volume > 1.5 * 20-period MA on 1d
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    highest_since_long = 0.0
    lowest_since_short = 0.0
    
    # Warmup: max of weekly EMA50 (50), ATR (14), volume MA (20)
    start_idx = max(50, 14, 20) + 4  # +4 to ensure 1 week of 1d data for weekly EMA50
    
    for i in range(start_idx, n):
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol = volume[i]
        ema50_val = weekly_ema50_aligned[i]
        r1_val = camarilla_r1_aligned[i]
        s1_val = camarilla_s1_aligned[i]
        vol_spike = volume_spike[i]
        atr_val = atr_1d_values[i]
        
        # Skip if any data not ready (NaN from alignment or calculation)
        if (np.isnan(ema50_val) or np.isnan(r1_val) or np.isnan(s1_val) or 
            np.isnan(atr_val) or np.isnan(volume_ma[i])):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Regime filter: bullish when price > weekly EMA50, bearish when price < weekly EMA50
        regime_bullish = close_val > ema50_val
        regime_bearish = close_val < ema50_val
        
        # Camarilla breakout conditions: price breaks R1/S1 with regime alignment + volume spike
        long_breakout = close_val > r1_val
        short_breakout = close_val < s1_val
        
        long_entry = regime_bullish and long_breakout and vol_spike
        short_entry = regime_bearish and short_breakout and vol_spike
        
        # Update highest/lowest for trailing stop (ATR-based)
        if position == 1:
            highest_since_long = max(highest_since_long, high_val)
        elif position == -1:
            lowest_since_short = min(lowest_since_short, low_val)
        elif position == 0:
            highest_since_long = 0.0
            lowest_since_short = 0.0
        
        # Exit conditions: ATR-based trailing stoploss
        long_exit = False
        short_exit = False
        if position == 1:
            # Long trailing stop: highest since entry - 2.0 * ATR
            stop_price = highest_since_long - 2.0 * atr_val
            long_exit = close_val < stop_price
        elif position == -1:
            # Short trailing stop: lowest since entry + 2.0 * ATR
            stop_price = lowest_since_short + 2.0 * atr_val
            short_exit = close_val > stop_price
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
            highest_since_long = high_val
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
            lowest_since_short = low_val
        elif long_exit:
            signals[i] = 0.0
            position = 0
            highest_since_long = 0.0
        elif short_exit:
            signals[i] = 0.0
            position = 0
            lowest_since_short = 0.0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_WeeklyTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0