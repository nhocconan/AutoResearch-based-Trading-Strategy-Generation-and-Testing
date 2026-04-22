#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 1-day EMA(34) trend filter and volume confirmation.
# Camarilla pivot levels provide short-term support/resistance zones. Breakouts above R1 or below S1
# signal momentum. The 1-day EMA(34) ensures trades align with the daily trend, reducing whipsaws.
# Volume confirmation (>1.5x 20-period average) filters false breakouts.
# This combination targets 20-40 trades per year, balancing opportunity and cost.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Camarilla pivot and EMA(34) (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 1-day EMA(34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Camarilla pivot levels from previous 1-day candle
    close_prev_1d = np.concatenate([[np.nan], close_1d[:-1]])
    high_prev_1d = np.concatenate([[np.nan], high_1d[:-1]])
    low_prev_1d = np.concatenate([[np.nan], low_1d[:-1]])
    range_1d = high_prev_1d - low_prev_1d
    # Avoid division by zero
    range_1d = np.where(range_1d == 0, 1e-10, range_1d)
    
    # Calculate Camarilla levels for each day, then align to 4h
    R1 = close_prev_1d + (range_1d * 1.1 / 12)
    S1 = close_prev_1d - (range_1d * 1.1 / 12)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Volume confirmation: 20-period average on 4h data
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(R1_aligned[i]) or
            np.isnan(S1_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: breakout above R1 + 1-day uptrend + volume confirmation
            if (close[i] > R1_aligned[i] and
                close[i] > ema_34_1d_aligned[i] and
                volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: breakout below S1 + 1-day downtrend + volume confirmation
            elif (close[i] < S1_aligned[i] and
                  close[i] < ema_34_1d_aligned[i] and
                  volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to opposite Camarilla level or trend reversal
            if position == 1:
                # Exit long: price returns to S1 or trend turns down
                if (close[i] < S1_aligned[i] or
                    close[i] < ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: price returns to R1 or trend turns up
                if (close[i] > R1_aligned[i] or
                    close[i] > ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_1dEMA34_VolumeConfirm"
timeframe = "4h"
leverage = 1.0