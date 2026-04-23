#!/usr/bin/env python3
"""
Hypothesis: 1h ADX + RSI regime filter with 4h Donchian breakout for direction.
- Use 4h Donchian(20) for trend direction (breakouts = momentum)
- Use 1h ADX(14) > 25 to filter for trending markets only
- Use 1h RSI(14) to avoid overbought/oversold extremes (long when RSI<70, short when RSI>30)
- Enter long when 4h Donchian upper break + ADX>25 + RSI<70
- Enter short when 4h Donchian lower break + ADX>25 + RSI>30
- Exit when opposite Donchian break occurs or ADX<20 (trend ends)
- Session filter: 08-20 UTC to avoid low-liquidity hours
- Position size: 0.20 (discrete to minimize fee churn)
- Target: 60-150 total trades over 4 years (15-37/year) to minimize fee drag on 1h
- Works in bull (trend continuation) and bear (avoids counter-trend via ADX filter)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1h indicators: ADX and RSI
    # ADX calculation
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[1:period+1])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    tr_smooth = wilders_smooth(tr, 14)
    plus_dm_smooth = wilders_smooth(plus_dm, 14)
    minus_dm_smooth = wilders_smooth(minus_dm, 14)
    
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smooth(dx, 14)
    
    # RSI calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Load 4h data ONCE before loop for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Donchian channels (20-period)
    donch_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align HTF Donchian levels to 1h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_4h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_4h, donch_low)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 34)  # Need 20 for Donchian, 34 for ADX/RSI (2*14+6)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Check session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: 4h Donchian upper break + ADX>25 (trending) + RSI<70 (not overbought)
            if close[i] > donch_high_aligned[i] and adx[i] > 25 and rsi[i] < 70:
                signals[i] = 0.20
                position = 1
            # Short: 4h Donchian lower break + ADX>25 (trending) + RSI>30 (not oversold)
            elif close[i] < donch_low_aligned[i] and adx[i] > 25 and rsi[i] > 30:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: 4h Donchian lower break OR ADX<20 (trend ending)
            if close[i] < donch_low_aligned[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: 4h Donchian upper break OR ADX<20 (trend ending)
            if close[i] > donch_high_aligned[i] or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_ADX_RSI_Regime_4hDonchian20_Breakout"
timeframe = "1h"
leverage = 1.0