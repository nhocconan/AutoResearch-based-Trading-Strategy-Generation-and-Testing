#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume confirmation
# Camarilla levels provide intraday support/resistance based on previous day's range.
# Breakout above R1 or below S1 with volume confirmation captures momentum.
# 1d EMA34 filter ensures alignment with daily trend, reducing counter-trend trades.
# Volume > 1.5x 20-period average filters false breakouts.
# Designed for 4h timeframe targeting 25-40 trades/year.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Camarilla levels and trend filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Camarilla: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, R2 = C + (H-L)*1.1/6, R1 = C + (H-L)*1.1/12
    # S1 = C - (H-L)*1.1/12, S2 = C - (H-L)*1.1/6, S3 = C - (H-L)*1.1/4, S4 = C - (H-L)*1.1/2
    # We use R1 and S1 for entries
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Avoid look-ahead: use previous day's data for today's levels
    camarilla_width = (prev_high - prev_low) * 1.1
    r1 = prev_close + camarilla_width / 12
    s1 = prev_close - camarilla_width / 12
    
    # Align Camarilla levels to 4h timeframe (wait for 1d bar to close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):  # Start after enough data for indicators
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Breakout above R1 + 1d uptrend + volume confirmation
            if (close[i] > r1_aligned[i] and  # breakout above R1
                close[i] > ema_34_1d_aligned[i] and  # price above 1d EMA (uptrend)
                volume[i] > 1.5 * vol_avg_20[i]):   # volume spike
                signals[i] = 0.25
                position = 1
            # Short: Breakout below S1 + 1d downtrend + volume confirmation
            elif (close[i] < s1_aligned[i] and   # breakout below S1
                  close[i] < ema_34_1d_aligned[i] and  # price below 1d EMA (downtrend)
                  volume[i] > 1.5 * vol_avg_20[i]):   # volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to opposite Camarilla level or trend reversal
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

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeConfirm"
timeframe = "4h"
leverage = 1.0