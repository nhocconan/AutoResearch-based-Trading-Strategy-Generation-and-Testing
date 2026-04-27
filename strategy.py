#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_4hTrend_1dVolume
Hypothesis: Camarilla pivot breakout on 1h with 4h trend filter and 1d volume confirmation.
- Camarilla levels: R1/S1 (inner support/resistance) from previous 1h OHLC.
- 4h trend: EMA50 on 4h (bullish = close > EMA50, bearish = close < EMA50).
- 1d volume: current volume > 1.5 * 20-day average volume.
- Long: price breaks above R1 + 4h bullish + 1d volume spike.
- Short: price breaks below S1 + 4h bearish + 1d volume spike.
- Exit: opposite Camarilla breakout or trend failure.
- Session filter: 08-20 UTC to avoid low-liquidity hours.
- Target: 15-35 trades/year (60-140 total over 4 years) on 1h.
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
    
    # Camarilla R1/S1 from previous bar (standard formula)
    # R1 = close + 1.1 * (high - low) / 12
    # S1 = close - 1.1 * (high - low) / 12
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    camarilla_range = prev_high - prev_low
    r1 = prev_close + 1.1 * camarilla_range / 12
    s1 = prev_close - 1.1 * camarilla_range / 12
    
    # 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    ema50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 1d volume MA(20) for volume spike filter
    df_1d = get_htf_data(prices, '1d')
    vol_ma20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    
    # Session filter: 08-20 UTC
    hour = pd.DatetimeIndex(prices['open_time']).hour
    session_mask = (hour >= 8) & (hour <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup
    start_idx = max(50, 20) + 1
    
    for i in range(start_idx, n):
        if (np.isnan(r1[i]) or np.isnan(s1[i]) or np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(vol_ma20_1d_aligned[i]) or not session_mask[i]):
            signals[i] = 0.0
            continue
        
        vol_spike = volume[i] > (1.5 * vol_ma20_1d_aligned[i])
        
        if position == 0:
            # Long entry: break above R1 + 4h bullish + 1d volume spike
            if (close[i] > r1[i] and close[i] > ema50_4h_aligned[i] and vol_spike):
                signals[i] = 0.20
                position = 1
            # Short entry: break below S1 + 4h bearish + 1d volume spike
            elif (close[i] < s1[i] and close[i] < ema50_4h_aligned[i] and vol_spike):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: break below S1 or 4h bearish
            if (close[i] < s1[i] or close[i] < ema50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: break above R1 or 4h bullish
            if (close[i] > r1[i] or close[i] > ema50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hTrend_1dVolume"
timeframe = "1h"
leverage = 1.0