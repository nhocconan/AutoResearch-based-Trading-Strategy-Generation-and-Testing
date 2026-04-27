#!/usr/bin/env python3
"""
1h_Camarilla_R3_S3_Breakout_4hEMA50_Trend_VolumeSpike_SessionFilter
Hypothesis: On 1h timeframe, use 4h Camarilla R3/S3 breakouts aligned with 4h EMA50 trend and volume spike.
Restrict entries to 08-20 UTC session to avoid low-liquidity hours. Uses discrete position sizing (0.20)
to minimize fee churn. Target: 15-35 trades/year per symbol to stay within fee drag limits.
Works in bull/bear by following 4h trend direction.
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
    
    # Calculate ATR for stoploss (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 4h Camarilla pivot levels (focus on R3/S3 for breakout entries)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    PP = (high_4h + low_4h + close_4h) / 3.0
    range_4h = high_4h - low_4h
    
    # Key levels: R3 and S3 for breakout entries
    R3 = PP + range_4h * 1.1 / 4.0
    S3 = PP - range_4h * 1.1 / 4.0
    
    # Align Camarilla levels to 1h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_4h, R3)
    S3_aligned = align_htf_to_ltf(prices, df_4h, S3)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    # Session filter: 08-20 UTC (already datetime64[ms] index)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need enough for ATR, EMA50 and volume average
    start_idx = max(100, 50, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_spike[i]) or np.isnan(atr[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_trend = ema_50_aligned[i]
        vol_spike = volume_spike[i]
        atr_val = atr[i]
        size = 0.20  # 20% position size
        
        if position == 0:
            # Flat - look for entry: breakout in direction of 4h trend with volume spike
            # Long: price breaks above R3 AND 4h trend is up (price > EMA50) AND volume spike
            # Short: price breaks below S3 AND 4h trend is down (price < EMA50) AND volume spike
            long_breakout = close_val > R3_aligned[i]
            short_breakout = close_val < S3_aligned[i]
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
            # Long - exit when price breaks below S3 (failed breakout) or ATR stoploss hit
            if close_val < S3_aligned[i] or close_val < entry_price - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit when price breaks above R3 (failed breakout) or ATR stoploss hit
            if close_val > R3_aligned[i] or close_val > entry_price + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "1h_Camarilla_R3_S3_Breakout_4hEMA50_Trend_VolumeSpike_SessionFilter"
timeframe = "1h"
leverage = 1.0