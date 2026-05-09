#!/usr/bin/env python3
"""
1d_WeeklyCVDivergence_Pullback
Hypothesis: Weekly CVD divergence with daily pullback entry for BTC/ETH.
- Uses weekly cumulative volume delta to detect smart money accumulation/distribution
- Enters on daily pullbacks in the direction of weekly CVD trend (buy dips in uptrend, sell rallies in downtrend)
- Filters with daily ADX > 25 to ensure trending market
- Weekly timeframe reduces noise, daily pullbacks provide good risk/reward
- Designed to work in both bull and bear markets by following weekly smart money flow
- Target: 15-30 trades/year to minimize fee drag
"""

name = "1d_WeeklyCVDivergence_Pullback"
timeframe = "1d"
leverage = 1.0

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
    volume = prices['volume'].values
    
    # Get weekly data for CVD calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate weekly CVD (Cumulative Volume Delta)
    # CVD = sum of (close - low) - (high - close) weighted by volume approximation
    # Using typical price as proxy for buy/sell pressure
    typical_price_1w = (high_1w + low_1w + close_1w) / 3.0
    # Buy pressure: close relative to low, Sell pressure: high relative to close
    buy_pressure = np.where(close_1w >= low_1w, (close_1w - low_1w) / (high_1w - low_1w + 1e-10), 0)
    sell_pressure = np.where(high_1w >= close_1w, (high_1w - close_1w) / (high_1w - low_1w + 1e-10), 0)
    # When high == low, avoid division by zero
    rng = high_1w - low_1w
    buy_pressure = np.where(rng == 0, 0.5, buy_pressure)
    sell_pressure = np.where(rng == 0, 0.5, sell_pressure)
    
    # CVD change per period: (buy - sell) * volume
    cvd_change = (buy_pressure - sell_pressure) * volume_1w
    cvd = np.cumsum(cvd_change)
    
    # Weekly CVD trend: slope over 4 weeks
    cvd_slope = np.full_like(cvd, np.nan)
    if len(cvd) >= 4:
        for i in range(3, len(cvd)):
            cvd_slope[i] = (cvd[i] - cvd[i-3]) / 3.0
    
    cvd_slope_aligned = align_htf_to_ltf(prices, df_1w, cvd_slope)
    
    # Get daily data for entry signals
    # ADX for trend strength filter
    # Calculate +DM, -DM, TR
    high_low = high - low
    high_prev_close = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    low_prev_close = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(high_low, np.maximum(high_prev_close, low_prev_close))
    
    plus_dm = np.where((high - np.concatenate([[high[0]], high[:-1]])) > 
                       (np.concatenate([[low[0]], low[:-1]]) - low), 
                       np.maximum(high - np.concatenate([[high[0]], high[:-1]]), 0), 0)
    minus_dm = np.where((np.concatenate([[low[0]], low[:-1]]) - low) > 
                        (high - np.concatenate([[high[0]], high[:-1]])), 
                        np.maximum(np.concatenate([[low[0]], low[:-1]]) - low, 0), 0)
    
    # Smooth with Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.mean(data[0:period])
            # Wilder's smoothing: today's value = (previous * (period-1) + current) / period
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    tr14 = wilders_smooth(tr, 14)
    plus_dm14 = wilders_smooth(plus_dm, 14)
    minus_dm14 = wilders_smooth(minus_dm, 14)
    
    # Avoid division by zero
    plus_di14 = np.where(tr14 != 0, 100 * plus_dm14 / tr14, 0)
    minus_di14 = np.where(tr14 != 0, 100 * minus_dm14 / tr14, 0)
    dx = np.where((plus_di14 + minus_di14) != 0, 
                  100 * np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14), 0)
    adx = wilders_smooth(dx, 14)
    
    # Daily pullback identification: price near recent swing points
    # For uptrend: look for pullbacks to recent lows
    # For downtrend: look for bounces to recent highs
    lookback = 10
    highest_high = np.full_like(high, np.nan)
    lowest_low = np.full_like(low, np.nan)
    
    for i in range(lookback, n):
        highest_high[i] = np.max(high[i-lookback:i])
        lowest_low[i] = np.min(low[i-lookback:i])
    
    # Pullback conditions
    # In uptrend (ADX rising and +DI > -DI): buy near recent lows
    # In downtrend (ADX rising and -DI > +DI): sell near recent highs
    pullback_long = np.zeros(n, dtype=bool)
    pullback_short = np.zeros(n, dtype=bool)
    
    for i in range(lookback, n):
        if not np.isnan(adx[i]) and adx[i] > 25:
            # Uptrend: +DI > -DI
            if plus_di14[i] > minus_di14[i]:
                # Pullback: price near recent low (within 1.5% of lowest low)
                if low[i] <= lowest_low[i] * 1.015:
                    pullback_long[i] = True
            # Downtrend: -DI > +DI
            elif minus_di14[i] > plus_di14[i]:
                # Bounce: price near recent high (within 1.5% of highest high)
                if high[i] >= highest_high[i] * 0.985:
                    pullback_short[i] = True
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after indicators are ready
    start_idx = max(lookback, 30)  # Need ADX and CVD slope ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(cvd_slope_aligned[i]) or np.isnan(adx[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: weekly CVD bullish (positive slope) AND daily pullback long signal
            if cvd_slope_aligned[i] > 0 and pullback_long[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: weekly CVD bearish (negative slope) AND daily pullback short signal
            elif cvd_slope_aligned[i] < 0 and pullback_short[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: weekly CVD turns bearish OR price breaks above recent high (momentum exhaustion)
            if cvd_slope_aligned[i] < 0 or (not np.isnan(highest_high[i]) and high[i] >= highest_high[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: weekly CVD turns bullish OR price breaks below recent low
            if cvd_slope_aligned[i] > 0 or (not np.isnan(lowest_low[i]) and low[i] <= lowest_low[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals