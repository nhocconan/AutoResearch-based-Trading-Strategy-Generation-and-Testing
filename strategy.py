#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 1d trend filter and volume confirmation
# Camarilla pivot levels (R1/S1) from 1-day data provide short-term support/resistance.
# Breakout above R1 or below S1 with volume confirmation indicates momentum.
# 1d EMA34 filter ensures trades align with daily trend (bullish if close > EMA34).
# Volume > 1.5x 20-day average confirms breakout strength.
# Designed for 4h timeframe targeting 20-40 trades/year to avoid fee drag.
# Works in bull markets by capturing upward breakouts and in bear markets by
# avoiding false breakdowns via trend filter and volume confirmation.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Camarilla pivots, trend filter, and volume (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla pivot levels for today using yesterday's OHLC
    # Camarilla: R1 = close + (high - low) * 1.1/12, S1 = close - (high - low) * 1.1/12
    # We need yesterday's data to calculate today's levels
    # Shift 1d data by 1 to get previous day's OHLC for today's levels
    if len(high_1d) < 2:
        return np.zeros(n)
    phigh = np.roll(high_1d, 1)  # previous day high
    plow = np.roll(low_1d, 1)    # previous day low
    pclose = np.roll(close_1d, 1) # previous day close
    # First day will have invalid data (rolled from last), but min_periods will handle
    phigh[0] = np.nan
    plow[0] = np.nan
    pclose[0] = np.nan
    
    camarilla_width = (phigh - plow) * 1.1 / 12
    r1 = pclose + camarilla_width  # Resistance 1
    s1 = pclose - camarilla_width  # Support 1
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d volume 20-period average for spike confirmation
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):  # Start after EMA34 warmup
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_avg_20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above R1 + 1d uptrend + volume spike
            if (close[i] > r1_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume[i] > 1.5 * vol_avg_20_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below S1 + 1d downtrend + volume spike
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume[i] > 1.5 * vol_avg_20_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price returns to opposite Camarilla level or trend reversal
            if position == 1:
                # Exit on return to S1 or trend reversal
                if (close[i] < s1_aligned[i] or 
                    close[i] < ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit on return to R1 or trend reversal
                if (close[i] > r1_aligned[i] or 
                    close[i] > ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0