#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Long when price breaks above Camarilla R3 AND close > 1d EMA34 AND volume > 2.0 * avg_volume(20)
# Short when price breaks below Camarilla S3 AND close < 1d EMA34 AND volume > 2.0 * avg_volume(20)
# Exit when price retouches Camarilla pivot point (PP) OR volume drops below average
# Uses discrete sizing 0.30 to balance return and risk
# Target: 100-200 total trades over 4 years (25-50/year) for 4h timeframe
# Camarilla levels provide precise intraday support/resistance in ranging markets
# 1d EMA34 filters for primary trend alignment to avoid counter-trend trades
# Volume spike confirms breakout strength and reduces false signals
# Works in bull markets (buying breakouts in uptrend) and bear markets (selling breakdowns in downtrend)

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike"
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
    
    # Get 1d data ONCE before loop for Camarilla calculation and EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need at least one completed daily bar for EMA34
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for the PREVIOUS daily bar
    # PP = (H + L + C) / 3
    # R3 = PP + (H - L) * 1.1 / 2
    # S3 = PP - (H - L) * 1.1 / 2
    pp = (high_1d + low_1d + close_1d) / 3
    r3 = pp + (high_1d - low_1d) * 1.1 / 2
    s3 = pp - (high_1d - low_1d) * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe (wait for completed daily bar)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    
    # Calculate 1d EMA34 for trend filter
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
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(pp_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3, above 1d EMA34, volume confirmation, in session
            if (close[i] > r3_aligned[i] and close[i-1] <= r3_aligned[i-1] and 
                close[i] > ema34_1d_aligned[i] and volume_confirm[i]):
                signals[i] = 0.30
                position = 1
            # Short: price breaks below S3, below 1d EMA34, volume confirmation, in session
            elif (close[i] < s3_aligned[i] and close[i-1] >= s3_aligned[i-1] and 
                  close[i] < ema34_1d_aligned[i] and volume_confirm[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price retouches pivot point (PP) OR volume drops below average
            if close[i] <= pp_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price retouches pivot point (PP) OR volume drops below average
            if close[i] >= pp_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals