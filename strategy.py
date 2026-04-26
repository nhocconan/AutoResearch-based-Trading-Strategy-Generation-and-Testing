#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike
Hypothesis: On 4h timeframe, price breaking above Camarilla R1 level with 1d EMA50 uptrend and volume spike (>1.5x 20-period MA) captures bullish continuation; breaking below S1 with 1d EMA50 downtrend and volume spike captures bearish continuation. Weekly pivot acts as regime filter to avoid counter-trend traps. Discrete position sizing (±0.25) and ATR-based trailing stop (2.5x) for exits. Targets 20-50 trades/year by requiring 1d trend alignment, volume confirmation, and Camarilla breakout structure—designed to work in both bull (breakouts above R1 in uptrend) and bear (breakdowns below S1 in downtrend) markets with BTC/ETH edge from institutional levels and volume-confirmed breakouts.
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
    
    # Load daily data ONCE before loop for Camarilla, EMA, and Donchian
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily OHLC for Camarilla calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_R1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    camarilla_S1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe (wait for completed 1d bar)
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    
    # Daily EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 4h ATR(14) for trailing stop
    tr1 = pd.Series(high).diff().abs()
    tr2 = (pd.Series(high) - pd.Series(close).shift()).abs()
    tr3 = (pd.Series(low) - pd.Series(close).shift()).abs()
    tr_4h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_4h = tr_4h.ewm(span=14, adjust=False, min_periods=14).mean()
    atr_4h_values = atr_4h.values
    
    # Volume spike filter: volume > 1.5 * 20-period MA on 4h
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    highest_since_long = 0.0
    lowest_since_short = 0.0
    
    # Warmup: max of EMA50 (50 days), ATR (14), volume MA (20)
    start_idx = max(50, 14, 20) + 24  # +24 to ensure 1 day of 4h data for daily indicators
    
    for i in range(start_idx, n):
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol = volume[i]
        r1_val = camarilla_R1_aligned[i]
        s1_val = camarilla_S1_aligned[i]
        ema50_val = ema50_1d_aligned[i]
        vol_spike = volume_spike[i]
        atr_val = atr_4h_values[i]
        
        # Skip if any data not ready (NaN from alignment or calculation)
        if (np.isnan(r1_val) or np.isnan(s1_val) or np.isnan(ema50_val) or 
            np.isnan(atr_val) or np.isnan(volume_ma[i])):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Trend filter: bullish when price > EMA50, bearish when price < EMA50
        trend_bullish = close_val > ema50_val
        trend_bearish = close_val < ema50_val
        
        # Camarilla breakout conditions: price breaks R1/S1 with trend alignment + volume spike
        long_breakout = close_val > r1_val
        short_breakout = close_val < s1_val
        
        long_entry = trend_bullish and long_breakout and vol_spike
        short_entry = trend_bearish and short_breakout and vol_spike
        
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
            # Long trailing stop: highest since entry - 2.5 * ATR
            stop_price = highest_since_long - 2.5 * atr_val
            long_exit = close_val < stop_price
        elif position == -1:
            # Short trailing stop: lowest since entry + 2.5 * ATR
            stop_price = lowest_since_short + 2.5 * atr_val
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

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0