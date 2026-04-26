#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_RegimeADX
Hypothesis: On 4h timeframe, price breaking Camarilla R1/S1 levels in the direction of 1d EMA34 trend with volume confirmation (>1.5x 20-period MA) and ADX>25 (trending regime) captures high-probability trend continuation moves. ADX filter reduces whipsaws in ranging markets, improving performance in both bull/bear regimes. Designed for 20-50 trades/year with discrete sizing (±0.30) and ATR-based trailing stop (2.0x) to minimize fee drag.
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
    
    # Load 1d data ONCE before loop for Camarilla calculation and EMA trend
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
    
    # Align Camarilla levels to 4h timeframe (wait for completed 1d bar)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # 1d EMA34 for trend filter
    close_series = pd.Series(close_1d)
    ema_34 = close_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # 4h ATR(20) for trailing stop
    tr1 = pd.Series(high).diff().abs()
    tr2 = (pd.Series(high) - pd.Series(close).shift()).abs()
    tr3 = (pd.Series(low) - pd.Series(close).shift()).abs()
    tr_4h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_4h = tr_4h.ewm(span=20, adjust=False, min_periods=20).mean()
    atr_4h_values = atr_4h.values
    
    # Volume spike filter: volume > 1.5 * 20-period MA on 4h
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    # ADX(14) for regime filter - trending when ADX > 25
    plus_dm = pd.Series(high).diff()
    minus_dm = pd.Series(low).diff().mul(-1)
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    tr_14 = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr_14.ewm(span=14, adjust=False, min_periods=14).mean()
    
    plus_di = 100 * (plus_dm.ewm(span=14, adjust=False, min_periods=14).mean() / atr_14)
    minus_di = 100 * (minus_dm.ewm(span=14, adjust=False, min_periods=14).mean() / atr_14)
    dx = (abs(plus_di - minus_di) / (abs(plus_di + minus_di))).replace([np.inf, -np.inf], 0).fillna(0) * 100
    adx = dx.ewm(span=14, adjust=False, min_periods=14).mean()
    adx_values = adx.values
    adx_trending = adx_values > 25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.30
    highest_since_long = 0.0
    lowest_since_short = 0.0
    
    # Warmup: max of EMA (34), ATR (20,14), volume MA (20) + time for 1d alignment
    start_idx = max(34, 20, 14, 20) + 6  # +6 to ensure 1d bar completion (4h -> 1d: 6 bars per day)
    
    for i in range(start_idx, n):
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol = volume[i]
        r1_val = camarilla_r1_aligned[i]
        s1_val = camarilla_s1_aligned[i]
        ema_val = ema_34_aligned[i]
        vol_spike = volume_spike[i]
        atr_val = atr_4h_values[i]
        adx_val = adx_trending[i]
        
        # Skip if any data not ready (NaN from alignment or calculation)
        if (np.isnan(r1_val) or np.isnan(s1_val) or np.isnan(ema_val) or 
            np.isnan(atr_val) or np.isnan(volume_ma[i]) or np.isnan(adx_val)):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Trend filter: bullish when price > EMA34, bearish when price < EMA34
        trend_bullish = close_val > ema_val
        trend_bearish = close_val < ema_val
        
        # Camarilla breakout conditions: price breaks R1/S1 with trend alignment + volume spike + ADX trending
        long_breakout = close_val > r1_val
        short_breakout = close_val < s1_val
        
        long_entry = trend_bullish and long_breakout and vol_spike and adx_val
        short_entry = trend_bearish and short_breakout and vol_spike and adx_val
        
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

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_RegimeADX"
timeframe = "4h"
leverage = 1.0