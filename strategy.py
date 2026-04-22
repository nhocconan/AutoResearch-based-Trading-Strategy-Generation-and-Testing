# 12H_Camarilla_R1_S1_Breakout_1dEMA34_Trend_Volume
# Target: 12h timeframe with daily CAMARILLA R1/S1 breakouts + EMA34 trend filter + volume confirmation
# Strategy: Breakouts of daily CAMARILLA R1/S1 on 12h candles with volume > 1.5x 20-period average and price above/below daily EMA34
# Exit: Opposite S1/R1 crossover for tight risk control
# Position size: 0.25 (25%) to balance risk and return
# Expected trades: 15-25 per year (60-100 total over 4 years) - within optimal range for 12h
# Works in bull/bear: Trend filter (EMA34) ensures alignment with daily trend, volume confirms breakout strength

#!/usr/bin/env python3
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
    
    # Load daily data for CAMARILLA and EMA (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's high, low, close for CAMARILLA pivot points
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate CAMARILLA pivot levels (R1/S1 only - tighter levels for more precise entries)
    pivot = (high_1d + low_1d + close_1d) / 3
    range_ = high_1d - low_1d
    r1 = close_1d + range_ * 1.1 / 12  # Resistance level 1
    s1 = close_1d - range_ * 1.1 / 12  # Support level 1
    
    # Trend filter: 1d EMA34 (more responsive than EMA50 for trend)
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume confirmation: 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R1 with volume AND above 1d EMA34 (uptrend)
            if (close[i] > r1_aligned[i] and volume[i] > 1.5 * vol_avg_20[i] and 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 with volume AND below 1d EMA34 (downtrend)
            elif (close[i] < s1_aligned[i] and volume[i] > 1.5 * vol_avg_20[i] and 
                  close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses back to opposite S1/R1 level (tighter stop)
            if position == 1:
                if not np.isnan(s1_aligned[i]) and close[i] < s1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if not np.isnan(r1_aligned[i]) and close[i] > r1_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12H_Camarilla_R1_S1_Breakout_1dEMA34_Trend_Volume"
timeframe = "12h"
leverage = 1.0