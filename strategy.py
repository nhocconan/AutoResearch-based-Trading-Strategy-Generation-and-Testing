#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 pivot breakout with 1d EMA34 trend filter and volume confirmation
# Camarilla pivot levels act as support/resistance where price often reverses or breaks out.
# Breakout of R1 (resistance 1) or S1 (support 1) with volume confirmation captures momentum.
# 1d EMA34 filter ensures alignment with daily trend, reducing counter-trend trades.
# Designed for 4h timeframe targeting 20-40 trades/year with low frequency to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Camarilla pivot and EMA trend (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels (R1, S1) from previous day
    # Formula: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan  # first value invalid
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    camarilla_width = 1.1 * (prev_high - prev_low) / 12
    r1 = prev_close + camarilla_width  # Resistance 1
    s1 = prev_close - camarilla_width  # Support 1
    
    # Align R1 and S1 to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: breakout above R1 + daily uptrend + volume confirmation
            if (close[i] > r1_aligned[i] and      # breakout above R1
                close[i] > ema_34_1d_aligned[i] and  # price above daily EMA (uptrend)
                volume[i] > 1.5 * vol_avg_20[i]):    # volume spike
                signals[i] = 0.25
                position = 1
            # Short: breakdown below S1 + daily downtrend + volume confirmation
            elif (close[i] < s1_aligned[i] and     # breakdown below S1
                  close[i] < ema_34_1d_aligned[i] and  # price below daily EMA (downtrend)
                  volume[i] > 1.5 * vol_avg_20[i]):    # volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to opposite pivot level or trend reversal
            if position == 1:
                # Exit long: price returns to S1 or trend turns down
                if (close[i] < s1_aligned[i] or 
                    close[i] < ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: price returns to R1 or trend turns up
                if (close[i] > r1_aligned[i] or 
                    close[i] > ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_Volume"
timeframe = "4h"
leverage = 1.0