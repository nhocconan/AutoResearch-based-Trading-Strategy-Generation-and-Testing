#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_Regime
Hypothesis: On 12h timeframe, price breaking Camarilla R1/S1 levels in the direction of 1d EMA34 trend with volume confirmation (>1.5x 20-period MA) and choppiness regime filter (CHOP > 61.8 for ranging market mean reversion) captures high-probability trades. The 1d EMA34 acts as a dynamic trend filter, Camarilla levels provide precise entry/exit zones, volume spike confirms institutional participation, and chop filter avoids whipsaws in strong trends. Designed for 12-37 trades/year with discrete sizing (±0.30) and ATR-based trailing stop (2.0x) to minimize fee drag and work in both bull/bear markets with BTC/ETH edge.
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
    
    # Load 1d data ONCE before loop for Camarilla calculation, EMA trend, and chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    daily_range = high_1d - low_1d
    camarilla_r1 = close_1d + daily_range * 1.1 / 12
    camarilla_s1 = close_1d - daily_range * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe (wait for completed 1d bar)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # 1d EMA34 for trend filter
    close_series = pd.Series(close_1d)
    ema_34 = close_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # 1d Choppiness Index for regime filter (CHOP > 61.8 = ranging market)
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    close_series_1d = pd.Series(close_1d)
    
    # True Range
    tr1 = high_series.diff().abs()
    tr2 = (high_series - close_series_1d.shift()).abs()
    tr3 = (low_series - close_series_1d.shift()).abs()
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr_1d.rolling(window=14, min_periods=14).sum()
    
    # Highest high and lowest low over 14 periods
    hh_14 = high_series.rolling(window=14, min_periods=14).max()
    ll_14 = low_series.rolling(window=14, min_periods=14).min()
    
    # Choppiness Index: CHOP = 100 * log10(sum(ATR14) / (HH14 - LL14)) / log10(14)
    chop_raw = 100 * np.log10(atr_1d / (hh_14 - ll_14)) / np.log10(14)
    chop_values = chop_raw.fillna(50).values  # fill NaN with neutral 50
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_values)
    
    # 12h ATR(20) for trailing stop
    tr1_12h = pd.Series(high).diff().abs()
    tr2_12h = (pd.Series(high) - pd.Series(close).shift()).abs()
    tr3_12h = (pd.Series(low) - pd.Series(close).shift()).abs()
    tr_12h = pd.concat([tr1_12h, tr2_12h, tr3_12h], axis=1).max(axis=1)
    atr_12h = tr_12h.ewm(span=20, adjust=False, min_periods=20).mean()
    atr_12h_values = atr_12h.values
    
    # Volume spike filter: volume > 1.5 * 20-period MA on 12h
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.30
    highest_since_long = 0.0
    lowest_since_short = 0.0
    
    # Warmup: max of EMA (34), ATR (20), volume MA (20), chop (14) + time for 1d alignment
    start_idx = max(34, 20, 20, 14) + 2  # +2 to ensure 1d bar completion (12h -> 1d: 2 bars per day)
    
    for i in range(start_idx, n):
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol = volume[i]
        r1_val = camarilla_r1_aligned[i]
        s1_val = camarilla_s1_aligned[i]
        ema_val = ema_34_aligned[i]
        chop_val = chop_aligned[i]
        vol_spike = volume_spike[i]
        atr_val = atr_12h_values[i]
        
        # Skip if any data not ready (NaN from alignment or calculation)
        if (np.isnan(r1_val) or np.isnan(s1_val) or np.isnan(ema_val) or 
            np.isnan(chop_val) or np.isnan(atr_val) or np.isnan(volume_ma[i])):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Regime filter: only trade in ranging market (CHOP > 61.8)
        in_ranging_regime = chop_val > 61.8
        
        # Trend filter: bullish when price > EMA34, bearish when price < EMA34
        trend_bullish = close_val > ema_val
        trend_bearish = close_val < ema_val
        
        # Camarilla breakout conditions: price breaks R1/S1 with trend alignment + volume spike + regime filter
        long_breakout = close_val > r1_val
        short_breakout = close_val < s1_val
        
        long_entry = trend_bullish and long_breakout and vol_spike and in_ranging_regime
        short_entry = trend_bearish and short_breakout and vol_spike and in_ranging_regime
        
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

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_Regime"
timeframe = "12h"
leverage = 1.0