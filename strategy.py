#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using daily Elder Ray Index (Bull Power/Bear Power) with 1d EMA13 trend filter and ATR-based volatility regime
# Long when Bull Power > 0 AND price > 1d EMA13 AND ATR(14) > ATR(50) (high volatility regime)
# Short when Bear Power < 0 AND price < 1d EMA13 AND ATR(14) > ATR(50) (high volatility regime)
# Exit when Bull/Bear Power crosses zero OR ATR(14) < ATR(50) (low volatility regime)
# Uses discrete sizing 0.25 to balance return and risk
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Elder Ray measures trend strength via power of bulls/bears relative to EMA
# 1d EMA13 filters for primary trend alignment to avoid counter-trend trades
# ATR regime filter ensures we only trade in sufficient volatility environments
# Works in bull markets (buying strength in uptrend) and bear markets (selling weakness in downtrend)

name = "6h_ElderRay_ATR_Regime_1dEMA13"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Elder Ray and EMA13
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough for EMA13 and ATR(50)
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA13 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema13_1d = close_1d_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Bull Power and Bear Power (Elder Ray)
    # Bull Power = High - EMA13
    # Bear Power = Low - EMA13
    bull_power = high_1d - ema13_1d
    bear_power = low_1d - ema13_1d
    
    # Align Elder Ray components to 6h timeframe (wait for completed daily bar)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    
    # Calculate ATR(14) and ATR(50) for volatility regime filter
    # True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
    prev_close_1d = np.append([np.nan], close_1d[:-1])
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - prev_close_1d)
    tr3 = np.abs(low_1d - prev_close_1d)
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate ATR using Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def calculate_atr(tr, period):
        atr = np.full_like(tr, np.nan)
        if len(tr) < period:
            return atr
        # First ATR is simple average
        atr[period-1] = np.nanmean(tr[:period])
        # Wilder's smoothing: ATR today = (ATR yesterday * (period-1) + TR today) / period
        for i in range(period, len(tr)):
            if not np.isnan(atr[i-1]):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr14_1d = calculate_atr(true_range, 14)
    atr50_1d = calculate_atr(true_range, 50)
    
    # Align ATR indicators to 6h timeframe
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    atr50_1d_aligned = align_htf_to_ltf(prices, df_1d, atr50_1d)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema13_1d_aligned[i]) or np.isnan(atr14_1d_aligned[i]) or 
            np.isnan(atr50_1d_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 (bulls in control) AND price > 1d EMA13 AND high volatility regime
            if (bull_power_aligned[i] > 0 and close[i] > ema13_1d_aligned[i] and 
                atr14_1d_aligned[i] > atr50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 (bears in control) AND price < 1d EMA13 AND high volatility regime
            elif (bear_power_aligned[i] < 0 and close[i] < ema13_1d_aligned[i] and 
                  atr14_1d_aligned[i] > atr50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power crosses zero OR low volatility regime
            if bull_power_aligned[i] <= 0 or atr14_1d_aligned[i] <= atr50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power crosses zero OR low volatility regime
            if bear_power_aligned[i] >= 0 or atr14_1d_aligned[i] <= atr50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals