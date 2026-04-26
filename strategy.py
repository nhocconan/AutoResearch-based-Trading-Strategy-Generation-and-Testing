#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike_v1
Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA50 trend filter and volume spike confirmation.
- Long when price breaks above Camarilla R1 AND 4h EMA50 uptrend AND volume > 1.5 * volume MA(20)
- Short when price breaks below Camarilla S1 AND 4h EMA50 downtrend AND volume > 1.5 * volume MA(20)
- Uses Camarilla levels from prior 1h bar (completed bar only) for structure-based breakouts
- 4h EMA50 filter ensures trading with higher timeframe trend to avoid counter-trend whipsaws
- Volume spike confirms institutional participation and reduces false breakouts
- Session filter (08-20 UTC) to avoid low-liquidity periods
- Fixed position size 0.20 to control risk and minimize fee churn
- Designed for 1h timeframe targeting 15-35 trades/year to avoid fee drag
- Exit on opposite Camarilla level (S1 for longs, R1 for shorts) or trend reversal
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels from prior 1h bar (completed bar only)
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # where C = close, H = high, L = low of prior completed bar
    prev_close = np.concatenate([[np.nan], close[:-1]])
    prev_high = np.concatenate([[np.nan], high[:-1]])
    prev_low = np.concatenate([[np.nan], low[:-1]])
    
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Load 4h data ONCE before loop for trend filter (HTF)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA50 for trend filter (needs completed 4h candle)
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    # Trend: 1 = uptrend (close > EMA50), -1 = downtrend (close < EMA50), 0 = neutral/invalid
    trend_4h = np.where(ema_50_4h_aligned > 0, 
                        np.where(close > ema_50_4h_aligned, 1, -1), 
                        0)
    
    # Volume spike: volume > 1.5 * volume MA(20)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # prices.index is DatetimeIndex
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for volume MA, 50 for EMA)
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or
            np.isnan(trend_4h[i]) or np.isnan(volume_spike[i]) or
            not session_filter[i]):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Camarilla breakout conditions with trend and volume confirmation
        if position == 0:
            # Long: Price breaks above Camarilla R1 AND 4h uptrend AND volume spike
            if close[i] > camarilla_r1[i] and trend_4h[i] == 1 and volume_spike[i]:
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below Camarilla S1 AND 4h downtrend AND volume spike
            elif close[i] < camarilla_s1[i] and trend_4h[i] == -1 and volume_spike[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.20
            # Exit: Price falls below Camarilla S1 OR 4h trend turns down
            if close[i] < camarilla_s1[i] or trend_4h[i] == -1:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            # Exit: Price rises above Camarilla R1 OR 4h trend turns up
            if close[i] > camarilla_r1[i] or trend_4h[i] == 1:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0