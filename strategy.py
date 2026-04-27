#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolFilter
Hypothesis: Camarilla R1/S1 breakout on 1h with 4h EMA20 trend filter and 1d volume confirmation.
Trading only in direction of 4h trend avoids counter-trend whipsaws. Volume spike on 1d confirms
institutional participation. Designed for 1h timeframe with target 60-150 trades over 4 years.
Uses discrete position sizing (0.20) to minimize fee churn while maintaining adequate exposure.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate ATR for Camarilla levels (20-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Camarilla levels from previous 1h bar's OHLC
    prev_close = np.concatenate([[close[0]], close[:-1]])
    prev_high = np.concatenate([[high[0]], high[:-1]])
    prev_low = np.concatenate([[low[0]], low[:-1]])
    
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Calculate 4h EMA20 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Calculate 1d volume average for filter (20-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need enough for ATR, EMA20, volume average
    start_idx = max(100, 20, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or
            np.isnan(ema_20_aligned[i]) or np.isnan(vol_avg_1d_aligned[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        vol_1d_val = volume_1d[i] if i < len(volume_1d) else volume_1d[-1]
        ema_trend = ema_20_aligned[i]
        vol_avg_1d_val = vol_avg_1d_aligned[i]
        atr_val = atr[i]
        size = 0.20  # 20% position size
        
        # Session filter: 08-20 UTC
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Flat - look for entry: breakout in direction of 4h trend with 1d volume confirmation
            # Long: price breaks above Camarilla R1 AND 4h trend is up (close > EMA20) AND 1d volume > 1.5 * avg
            # Short: price breaks below Camarilla S1 AND 4h trend is down (close < EMA20) AND 1d volume > 1.5 * avg
            long_breakout = close_val > camarilla_r1[i]
            short_breakout = close_val < camarilla_s1[i]
            trend_up = close_val > ema_trend
            trend_down = close_val < ema_trend
            vol_spike = vol_1d_val > (1.5 * vol_avg_1d_val)
            
            if long_breakout and trend_up and vol_spike:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_breakout and trend_down and vol_spike:
                signals[i] = -size
                position = -1
                entry_price = close_val
        elif position == 1:
            # Long - exit when price breaks below Camarilla S1 (failed breakout) or ATR stoploss hit
            if close_val < camarilla_s1[i] or close_val < entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price breaks above Camarilla R1 (failed breakout) or ATR stoploss hit
            if close_val > camarilla_r1[i] or close_val > entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolFilter"
timeframe = "1h"
leverage = 1.0