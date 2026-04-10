#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1d trend filter and volume confirmation
# - Long when price breaks above Camarilla R4 (1d) AND 1d close > SMA(50) (bullish trend) AND 6h volume > 2.0x 20-bar avg
# - Short when price breaks below Camarilla S4 (1d) AND 1d close < SMA(50) (bearish trend) AND 6h volume > 2.0x 20-bar avg
# - Exit when price reverts to Camarilla PP (pivot point) or opposite S1/R1 level
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Camarilla levels provide institutional support/resistance; breakouts capture momentum
# - 1d SMA(50) filter ensures alignment with daily trend to avoid counter-trend trades
# - Volume spike (2.0x) confirms institutional participation in breakout
# - Target: 12-30 trades/year on 6h timeframe (50-120 total over 4 years)
# - Works in both bull and bear markets: breakouts work in trends, mean reversion exits work in ranges

name = "6h_1d_camarilla_breakout_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d Camarilla pivot levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point (PP) and Camarilla levels
    pp = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    r4 = pp + range_1d * 1.1 / 2.0
    r3 = pp + range_1d * 1.1 / 4.0
    r2 = pp + range_1d * 1.1 / 6.0
    r1 = pp + range_1d * 1.1 / 12.0
    s1 = pp - range_1d * 1.1 / 12.0
    s2 = pp - range_1d * 1.1 / 6.0
    s3 = pp - range_1d * 1.1 / 4.0
    s4 = pp - range_1d * 1.1 / 2.0
    
    # Pre-compute 1d trend filter: close > SMA(50) for bullish, close < SMA(50) for bearish
    sma_50 = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    trend_bullish = close_1d > sma_50
    trend_bearish = close_1d < sma_50
    
    # Align 1d indicators to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    trend_bullish_aligned = align_htf_to_ltf(prices, df_1d, trend_bullish)
    trend_bearish_aligned = align_htf_to_ltf(prices, df_1d, trend_bearish)
    
    # Pre-compute 6h volume confirmation: > 2.0x 20-period average
    volume = prices['volume'].values
    volume_20_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(pp_aligned[i]) or
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(trend_bullish_aligned[i]) or np.isnan(trend_bearish_aligned[i]) or
            np.isnan(vol_spike[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above R4 AND 1d bullish trend AND volume spike
            if (prices['close'].iloc[i] > r4_aligned[i] and 
                trend_bullish_aligned[i] and 
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below S4 AND 1d bearish trend AND volume spike
            elif (prices['close'].iloc[i] < s4_aligned[i] and 
                  trend_bearish_aligned[i] and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to mean reversion levels
            # Exit conditions: price reverts to PP or reaches opposite S1/R1 level
            if position == 1:  # Long position
                # Exit when price returns to PP or drops to S1 (mean reversion)
                exit_signal = (prices['close'].iloc[i] <= pp_aligned[i] or 
                              prices['close'].iloc[i] <= s1_aligned[i])
            else:  # Short position
                # Exit when price returns to PP or rises to R1 (mean reversion)
                exit_signal = (prices['close'].iloc[i] >= pp_aligned[i] or 
                              prices['close'].iloc[i] >= r1_aligned[i])
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals