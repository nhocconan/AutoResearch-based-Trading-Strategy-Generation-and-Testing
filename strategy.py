#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike
Hypothesis: On 12h timeframe, Camarilla R1/S1 breakout in the direction of daily EMA34 trend (price > EMA34 = bullish, price < EMA34 = bearish) with volume confirmation (>1.5x 20-period MA) captures high-probability trend continuation moves. Daily EMA34 acts as dynamic trend filter and regime identifier. Camarilla R1/S1 levels provide precise intraday support/resistance derived from prior day's range. Discrete position sizing (±0.25) and ATR-based trailing stop (2.0x) for exits. Targets 12-30 trades/year by requiring daily regime alignment, volume confirmation, and Camarilla breakout structure—designed to work in both bull (breakouts above daily EMA34) and bear (breakdowns below daily EMA34) markets with BTC/ETH edge from institutional Camarilla levels and volume-confirmed breakouts.
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
    
    # Load daily data ONCE before loop for EMA34 trend and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Daily EMA34 for trend direction
    close_1d = pd.Series(df_1d['close'].values)
    ema34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Daily Camarilla levels: based on prior day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R1, S1, R3, S3, R4, S4
    # R4 = Close + ((High - Low) * 1.500)
    # R3 = Close + ((High - Low) * 1.250)
    # R2 = Close + ((High - Low) * 1.166)
    # R1 = Close + ((High - Low) * 1.083)
    # S1 = Close - ((High - Low) * 1.083)
    # S2 = Close - ((High - Low) * 1.166)
    # S3 = Close - ((High - Low) * 1.250)
    # S4 = Close - ((High - Low) * 1.500)
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_hl = high_1d - low_1d
    camarilla_r1 = close_1d + (range_hl * 1.083)
    camarilla_s1 = close_1d - (range_hl * 1.083)
    camarilla_r3 = close_1d + (range_hl * 1.250)
    camarilla_s3 = close_1d - (range_hl * 1.250)
    
    # Align Camarilla levels to 12h timeframe (wait for completed 1d bar)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # 12h ATR(14) for trailing stop
    tr1 = pd.Series(high).diff().abs()
    tr2 = (pd.Series(high) - pd.Series(close).shift()).abs()
    tr3 = (pd.Series(low) - pd.Series(close).shift()).abs()
    tr_12h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_12h = tr_12h.ewm(span=14, adjust=False, min_periods=14).mean()
    atr_12h_values = atr_12h.values
    
    # Volume spike filter: volume > 1.5 * 20-period MA on 12h
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    highest_since_long = 0.0
    lowest_since_short = 0.0
    
    # Warmup: max of EMA34 (34), ATR (14), volume MA (20), daily data needs 1 day
    start_idx = max(34, 14, 20) + 2  # +2 to ensure 1 day of 12h data for daily alignment
    
    for i in range(start_idx, n):
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol = volume[i]
        ema34_val = ema34_1d_aligned[i]
        r1_val = camarilla_r1_aligned[i]
        s1_val = camarilla_s1_aligned[i]
        r3_val = camarilla_r3_aligned[i]
        s3_val = camarilla_s3_aligned[i]
        vol_spike = volume_spike[i]
        atr_val = atr_12h_values[i]
        
        # Skip if any data not ready (NaN from alignment or calculation)
        if (np.isnan(ema34_val) or np.isnan(r1_val) or np.isnan(s1_val) or 
            np.isnan(atr_val) or np.isnan(volume_ma[i])):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Regime filter: bullish when price > daily EMA34, bearish when price < daily EMA34
        regime_bullish = close_val > ema34_val
        regime_bearish = close_val < ema34_val
        
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

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0