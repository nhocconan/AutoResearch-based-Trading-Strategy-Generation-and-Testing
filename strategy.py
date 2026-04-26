#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike
Hypothesis: On 1h timeframe, price breaking Camarilla R1/S1 levels in the direction of 4h EMA50 trend with volume confirmation (>1.3x 20-period MA) captures high-probability trend continuation moves. The 4h EMA50 acts as a dynamic trend filter, while Camarilla levels provide precise entry/exit zones. Volume spike confirms institutional participation. Session filter (08-20 UTC) reduces noise. Designed for 15-37 trades/year with discrete sizing (±0.20) and ATR-based trailing stop (2.5x) to minimize fee drag and work in both bull/bear markets with BTC/ETH edge.
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
    open_time = prices['open_time'].values
    
    # Precompute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for Camarilla calculation and EMA trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous 4h bar
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla levels: based on previous bar's range
    daily_range = high_4h - low_4h
    camarilla_r1 = close_4h + daily_range * 1.1 / 12
    camarilla_s1 = close_4h - daily_range * 1.1 / 12
    
    # Align Camarilla levels to 1h timeframe (wait for completed 4h bar)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    
    # 4h EMA50 for trend filter
    close_series = pd.Series(close_4h)
    ema_50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    
    # 1h ATR(14) for trailing stop
    tr1 = pd.Series(high).diff().abs()
    tr2 = (pd.Series(high) - pd.Series(close).shift()).abs()
    tr3 = (pd.Series(low) - pd.Series(close).shift()).abs()
    tr_1h = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1h = tr_1h.ewm(span=14, adjust=False, min_periods=14).mean()
    atr_1h_values = atr_1h.values
    
    # Volume spike filter: volume > 1.3 * 20-period MA on 1h
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.20
    highest_since_long = 0.0
    lowest_since_short = 0.0
    
    # Warmup: max of EMA (50), ATR (14), volume MA (20)
    start_idx = max(50, 14, 20) + 4  # +4 to ensure 4h bar completion (1h -> 4h: 4 bars)
    
    for i in range(start_idx, n):
        if not in_session[i]:
            # Outside session: flatten position
            signals[i] = 0.0
            position = 0
            highest_since_long = 0.0
            lowest_since_short = 0.0
            continue
            
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol = volume[i]
        r1_val = camarilla_r1_aligned[i]
        s1_val = camarilla_s1_aligned[i]
        ema_val = ema_50_aligned[i]
        vol_spike = volume_spike[i]
        atr_val = atr_1h_values[i]
        
        # Skip if any data not ready (NaN from alignment or calculation)
        if (np.isnan(r1_val) or np.isnan(s1_val) or np.isnan(ema_val) or 
            np.isnan(atr_val) or np.isnan(volume_ma[i])):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Trend filter: bullish when price > EMA50, bearish when price < EMA50
        trend_bullish = close_val > ema_val
        trend_bearish = close_val < ema_val
        
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

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike"
timeframe = "1h"
leverage = 1.0