#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot (R4/S1) breakout with daily trend filter and volume confirmation.
# Uses daily pivot levels to identify key support/resistance. Long when price breaks above R4 in daily uptrend,
# short when breaks below S1 in daily downtrend. Volume spike confirms breakout strength.
# Designed for 4h timeframe to capture multi-day swings with low frequency (~25-40 trades/year).
# Target: 100-160 total trades over 4 years to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-day data for Camarilla pivot levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels (using prior day's data)
    # Pivot = (H + L + C) / 3
    # R4 = C + ((H - L) * 1.1 / 2)
    # S1 = C - ((H - L) * 1.1 / 2)
    pivot_1d = (np.roll(high_1d, 1) + np.roll(low_1d, 1) + np.roll(close_1d, 1)) / 3
    r4_1d = close_1d + ((high_1d - low_1d) * 1.1 / 2)
    s1_1d = close_1d - ((high_1d - low_1d) * 1.1 / 2)
    
    # Trend filter: price above/below daily EMA34
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_bullish = close_1d > ema34_1d
    trend_bearish = close_1d < ema34_1d
    
    # Volume spike filter (24-period on 4h)
    vol_ma24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_spike = volume > 2.0 * vol_ma24
    
    # Align indicators to 4-hour timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    trend_bullish_aligned = align_htf_to_ltf(prices, df_1d, trend_bullish.astype(float))
    trend_bearish_aligned = align_htf_to_ltf(prices, df_1d, trend_bearish.astype(float))
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(r4_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(trend_bullish_aligned[i]) or np.isnan(trend_bearish_aligned[i]) or
            np.isnan(vol_ma24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R4 + daily uptrend + volume spike
            if (close[i] > r4_aligned[i] and 
                trend_bullish_aligned[i] > 0.5 and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + daily downtrend + volume spike
            elif (close[i] < s1_aligned[i] and 
                  trend_bearish_aligned[i] > 0.5 and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price breaks opposite Camarilla level or trend changes
            if position == 1:
                if (close[i] < s1_aligned[i] or trend_bullish_aligned[i] <= 0.5):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if (close[i] > r4_aligned[i] or trend_bearish_aligned[i] <= 0.5):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R4_S1_Breakout_DailyEMA34_Trend_Volume_Spike"
timeframe = "4h"
leverage = 1.0