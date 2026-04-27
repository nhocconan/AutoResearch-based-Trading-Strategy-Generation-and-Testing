#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike_v1
Hypothesis: Camarilla R1/S1 breakout on 1h with 4h EMA50 trend filter and volume confirmation.
Breakouts above/below Camarilla R1/S1 levels capture strong momentum moves. Trend filter ensures
we only trade in direction of 4h trend to avoid counter-trend whipsaws. Volume spike confirms
breakout authenticity. Designed for 1h timeframe with target 60-150 trades over 4 years.
Uses discrete position sizing (0.20) to balance return and drawdown. Works in both bull
and bear markets by aligning with intermediate-term 4h trend.
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
    
    # Calculate 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need enough for ATR, EMA50 and volume average
    start_idx = max(100, 50, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_spike[i]) or np.isnan(atr[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_trend = ema_50_aligned[i]
        vol_spike = volume_spike[i]
        atr_val = atr[i]
        size = 0.20  # 20% position size
        
        if position == 0:
            # Flat - look for entry: breakout in direction of 4h trend with volume spike
            # Long: price breaks above Camarilla R1 AND 4h trend is up (close > EMA50) AND volume spike
            # Short: price breaks below Camarilla S1 AND 4h trend is down (close < EMA50) AND volume spike
            long_breakout = close_val > camarilla_r1[i]
            short_breakout = close_val < camarilla_s1[i]
            trend_up = close_val > ema_trend
            trend_down = close_val < ema_trend
            
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
            if close_val < camarilla_s1[i] or close_val < entry_price - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price breaks above Camarilla R1 (failed breakout) or ATR stoploss hit
            if close_val > camarilla_r1[i] or close_val > entry_price + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0