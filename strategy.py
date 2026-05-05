#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume spike confirmation
# Long when price breaks above Camarilla R1 (from previous 1d) AND close > 1d EMA34 AND volume > 2.0 * avg_volume(20)
# Short when price breaks below Camarilla S1 (from previous 1d) AND close < 1d EMA34 AND volume > 2.0 * avg_volume(20)
# Exit when price retouches Camarilla pivot point (PP) from previous 1d OR volume drops below average
# Uses discrete sizing 0.30 to balance return and risk
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# Camarilla levels provide high-probability intraday reversal/breakout points
# 1d EMA34 filters for primary trend alignment to avoid counter-trend trades
# Volume spike confirms breakout strength and reduces false signals
# Works in bull markets (buying breakouts in uptrend) and bear markets (selling breakdowns in downtrend)

name = "4h_Camarilla_R1S1_Breakout_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Camarilla levels and EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need enough for EMA34
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for previous 1d
    # Pivot Point (PP) = (High + Low + Close) / 3
    pp = (high_1d + low_1d + close_1d) / 3.0
    # Resistance 1 (R1) = Close + 1.1 * (High - Low) / 12
    r1 = close_1d + 1.1 * (high_1d - low_1d) / 12.0
    # Support 1 (S1) = Close - 1.1 * (High - Low) / 12
    s1 = close_1d - 1.1 * (high_1d - low_1d) / 12.0
    
    # Align Camarilla levels to 4h timeframe (wait for completed 1d bar)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Calculate 1d EMA34
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1, above 1d EMA34, volume confirmation, in session
            if (close[i] > r1_aligned[i] and close[i] > ema34_1d_aligned[i] and volume_confirm[i]):
                signals[i] = 0.30
                position = 1
            # Short: price breaks below S1, below 1d EMA34, volume confirmation, in session
            elif (close[i] < s1_aligned[i] and close[i] < ema34_1d_aligned[i] and volume_confirm[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price retouches PP OR volume drops below average
            if (close[i] <= pp_aligned[i]) or (volume[i] < avg_volume_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price retouches PP OR volume drops below average
            if (close[i] >= pp_aligned[i]) or (volume[i] < avg_volume_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals