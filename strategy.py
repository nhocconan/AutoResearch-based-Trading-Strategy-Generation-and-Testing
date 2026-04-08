# 12h_1d_weekly_pivot_momentum_v1
# Hypothesis: Trade weekly pivot point breakouts on 12h with 1d momentum confirmation.
# Uses weekly pivot levels (R1/R2/S1/S2) from prior week as key support/resistance.
# In bull markets: buy break above R1 with 1d bullish momentum.
# In bear markets: sell break below S1 with 1d bearish momentum.
# Momentum confirmed by 1d RSI > 50 (bullish) or < 50 (bearish).
# Volume surge (1.5x 20-period average) confirms breakout strength.
# Target: 12-37 trades/year with strict entry conditions to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_weekly_pivot_momentum_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d momentum: RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Weekly pivot points from 1d data (prior week's OHLC)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot: (Prior Week High + Low + Close) / 3
    # We need to group daily data into weeks
    # Simple approach: use rolling window of 5 days (approximate week)
    # But better: use actual weekly data from 1w timeframe if available
    # Since we're instructed to use 1d/1w as HTF, let's use 1w for pivot
    
    # Actually, let's use 1w data directly for weekly pivot
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        # Fallback to 1d approximation if 1w not available
        # Use 5-day rolling window for weekly OHLC
        week_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().shift(1)  # Prior week
        week_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().shift(1)
        week_close = pd.Series(close_1d).rolling(window=5, min_periods=5).last().shift(1)
    else:
        # Use actual weekly data
        week_high = df_1w['high'].values
        week_low = df_1w['low'].values
        week_close = df_1w['close'].values
        # Need to align weekly to daily then to 12h
        # First align weekly to 1d
        week_high_1d = align_htf_to_ltf(df_1d, df_1w, week_high)
        week_low_1d = align_htf_to_ltf(df_1d, df_1w, week_low)
        week_close_1d = align_htf_to_ltf(df_1d, df_1w, week_close)
        # Then align 1d to 12h
        week_high = align_htf_to_ltf(prices, df_1d, week_high_1d)
        week_low = align_htf_to_ltf(prices, df_1d, week_low_1d)
        week_close = align_htf_to_ltf(prices, df_1d, week_close_1d)
    
    # Calculate pivot points
    pivot = (week_high + week_low + week_close) / 3.0
    r1 = 2 * pivot - week_low
    s1 = 2 * pivot - week_high
    r2 = pivot + (week_high - week_low)
    s2 = pivot - (week_high - week_low)
    
    # Align weekly pivot levels to 12h timeframe
    # (already aligned if we used the 1w->1d->12h path above)
    
    # Volume confirmation: 12h volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 100  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi[i]) or np.isnan(pivot[i]) or np.isnan(r1[i]) or np.isnan(s1[i]) or
            np.isnan(r2[i]) or np.isnan(s2[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 1.5 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: Price below S1 OR RSI < 40 (loss of momentum)
            if close[i] < s1[i] or rsi[i] < 40:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price above R1 OR RSI > 60 (loss of momentum)
            if close[i] > r1[i] or rsi[i] > 60:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Break above R1 with 1d bullish momentum (RSI > 50) and volume surge
            if (close[i] > r1[i] and  # Break above R1
                rsi[i] > 50 and      # 1d bullish momentum
                vol_surge):
                position = 1
                signals[i] = 0.25
            # Short entry: Break below S1 with 1d bearish momentum (RSI < 50) and volume surge
            elif (close[i] < s1[i] and    # Break below S1
                  rsi[i] < 50 and         # 1d bearish momentum
                  vol_surge):
                position = -1
                signals[i] = -0.25
    
    return signals