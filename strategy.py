# 1h_Camarilla_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: Camarilla pivot levels (R3/S3) on daily timeframe act as strong support/resistance.
# Breakouts above R3 in a daily uptrend or below S3 in a daily downtrend, confirmed by volume,
# provide high-probability trades. Uses 1h for entry timing and daily for trend direction.
# Works in bull markets (breakouts in uptrend) and bear markets (breakdowns in downtrend).
# Target: 60-150 total trades over 4 years = 15-37/year for 1h.

#!/usr/bin/env python3
name = "1h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    # Using previous day's high, low, close (as Camarilla uses)
    phigh = df_1d['high'].shift(1).values  # Previous day high
    plow = df_1d['low'].shift(1).values    # Previous day low
    pclose = df_1d['close'].shift(1).values # Previous day close
    
    # Camarilla R3 and S3 levels
    camarilla_r3 = pclose + (phigh - plow) * 1.1 / 2
    camarilla_s3 = pclose - (phigh - plow) * 1.1 / 2
    
    # Align Camarilla levels to 1h timeframe (wait for daily close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Daily EMA34 trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: current volume > 1.5x 24-period average (for 1h, ~1 day)
    vol_ma_24 = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma_24[i] = np.mean(volume[i-24:i])
    vol_filter = volume > (1.5 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 6  # ~6 hours for 1h to reduce trades
    
    start_idx = max(100, 24, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine daily trend direction
        trend_up = close > ema_34_1d_aligned[i]
        trend_down = close < ema_34_1d_aligned[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Price breaks above Camarilla R3 with volume in uptrend
            if (close[i] > camarilla_r3_aligned[i] and 
                trend_up[i] and 
                vol_filter[i]):
                signals[i] = 0.20
                position = 1
                bars_since_last_trade = 0
            # Short: Price breaks below Camarilla S3 with volume in downtrend
            elif (close[i] < camarilla_s3_aligned[i] and 
                  trend_down[i] and 
                  vol_filter[i]):
                signals[i] = -0.20
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Price falls below Camarilla S3 or trend changes
            if close[i] < camarilla_s3_aligned[i] or not trend_up[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: Price rises above Camarilla R3 or trend changes
            if close[i] > camarilla_r3_aligned[i] or not trend_down[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.20
    
    return signals

# Hypothesis: Camarilla pivot levels (R3/S3) on daily timeframe act as strong support/resistance.
# Breakouts above R3 in a daily uptrend or below S3 in a daily downtrend, confirmed by volume,
# provide high-probability trades. Uses 1h for entry timing and daily for trend direction.
# Works in bull markets (breakouts in uptrend) and bear markets (breakdowns in downtrend).
# Target: 60-150 total trades over 4 years = 15-37/year for 1h.