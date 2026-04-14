#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + Weekly EMA Filter + Volume Spike
# Elder Ray measures bull/bear power: bull_power = high - EMA(13), bear_power = low - EMA(13)
# Combines with weekly EMA trend filter (avoid counter-trend trades) and volume confirmation
# Works in bull/bear by only taking Elder Ray signals aligned with weekly trend
# Target: 50-150 total trades over 4 years (12-37/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # EMA 13 for Elder Ray calculation
    close_series = pd.Series(close)
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Weekly trend filter: EMA 21 on weekly data
    df_1w = get_htf_data(prices, '1w')
    ema21_1w = pd.Series(df_1w['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    
    # Volume confirmation: volume > 1.5x average volume (20-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 35  # for EMA13 and volume average
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema21_1w_aligned[i]) or np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: only take bullish signals when price > weekly EMA21,
        # only take bearish signals when price < weekly EMA21
        weekly_uptrend = close[i] > ema21_1w_aligned[i]
        weekly_downtrend = close[i] < ema21_1w_aligned[i]
        
        if position == 0:
            # Long: bullish Elder Ray + weekly uptrend + volume spike
            if bull_power[i] > 0 and weekly_uptrend and volume[i] > 1.5 * avg_vol[i]:
                position = 1
                signals[i] = position_size
            # Short: bearish Elder Ray + weekly downtrend + volume spike
            elif bear_power[i] < 0 and weekly_downtrend and volume[i] > 1.5 * avg_vol[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: bearish Elder Ray or weekly trend turns down
            if bear_power[i] < 0 or not weekly_uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: bullish Elder Ray or weekly trend turns up
            if bull_power[i] > 0 or not weekly_downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_ElderRay_WeeklyEMA_VolumeFilter"
timeframe = "6h"
leverage = 1.0