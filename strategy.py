#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_RegimeFilter_v1
Hypothesis: 6-hour Elder Ray (Bull/Bear Power) with 1-day EMA13 trend filter and regime detection via ADX(14) < 20 for mean reversion.
Targets 12-37 trades/year by requiring: 1) Bull Power > 0 and Bear Power < 0 for momentum confirmation,
2) price > 1d EMA13 for uptrend bias (or < for downtrend), 3) ADX < 20 indicating ranging market for mean reversion entries.
Uses 6h timeframe to balance trade frequency and fee drag while capturing reversals in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 1d data for EMA13 trend filter (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    ema_13_1d = pd.Series(df_1d['close'].values).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # 1d data for ADX(14) regime filter (loaded ONCE)
    # Calculate ADX components: +DM, -DM, TR
    plus_dm = np.zeros(len(df_1d))
    minus_dm = np.zeros(len(df_1d))
    tr = np.zeros(len(df_1d))
    
    for i in range(1, len(df_1d)):
        high_diff = df_1d['high'].iloc[i] - df_1d['high'].iloc[i-1]
        low_diff = df_1d['low'].iloc[i-1] - df_1d['low'].iloc[i]
        plus_dm[i] = max(high_diff, 0) if high_diff > low_diff else 0
        minus_dm[i] = max(low_diff, 0) if low_diff > high_diff else 0
        tr[i] = max(df_1d['high'].iloc[i] - df_1d['low'].iloc[i],
                    abs(df_1d['high'].iloc[i] - df_1d['close'].iloc[i-1]),
                    abs(df_1d['low'].iloc[i] - df_1d['close'].iloc[i-1]))
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period = 14
    if len(tr) >= period:
        atr = wilder_smooth(tr, period)
        plus_di = 100 * wilder_smooth(plus_dm, period) / atr
        minus_di = 100 * wilder_smooth(minus_dm, period) / atr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = wilder_smooth(dx, period)
    else:
        adx = np.full_like(tr, np.nan)
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13_1d_aligned
    bear_power = low - ema_13_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for ADX calculation
    start_idx = 14 + 13  # ADX period + EMA warmup
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(ema_13_1d_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        # Regime filter: ADX < 20 indicates ranging market (mean reversion favorable)
        ranging = adx_aligned[i] < 20
        
        # Trend filter: price relative to 1d EMA13
        uptrend = curr_close > ema_13_1d_aligned[i]
        downtrend = curr_close < ema_13_1d_aligned[i]
        
        if position == 0:
            # Look for entry signals in ranging market
            # Long: Bull Power > 0 (strong bulls) and Bear Power < 0 (weak bears) in uptrend bias
            # Short: Bear Power < 0 (strong bears) and Bull Power > 0 (weak bulls) in downtrend bias
            long_signal = ranging and bull_power[i] > 0 and bear_power[i] < 0 and uptrend
            short_signal = ranging and bear_power[i] < 0 and bull_power[i] > 0 and downtrend
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit conditions
            # Exit if Bear Power becomes positive (bulls weakening) or trend changes
            if bear_power[i] >= 0 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit conditions
            # Exit if Bull Power becomes negative (bears weakening) or trend changes
            if bull_power[i] <= 0 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_BullBearPower_RegimeFilter_v1"
timeframe = "6h"
leverage = 1.0