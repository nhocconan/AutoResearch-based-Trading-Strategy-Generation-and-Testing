#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot (R1/S1) breakout with 1d EMA trend filter and volume confirmation
# Camarilla pivot levels provide high-probability reversal points in ranging markets.
# 1d EMA filter ensures alignment with daily trend to avoid counter-trend trades.
# Volume confirmation (>1.8x 20-period average) filters false breakouts.
# Designed for 12h timeframe targeting 12-30 trades/year with strong risk-adjusted returns.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Camarilla pivot and EMA (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels (R1, S1) from previous day
    # Camarilla: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    hl_range = high_1d - low_1d
    camarilla_r1 = close_1d + 1.1 * hl_range / 12
    camarilla_s1 = close_1d - 1.1 * hl_range / 12
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 12h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):
        # Skip if data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Camarilla R1 + 1d uptrend + volume confirmation
            if (close[i] > camarilla_r1_aligned[i] and      # breakout above R1
                close[i] > ema_34_1d_aligned[i] and        # price above 1d EMA (uptrend)
                volume[i] > 1.8 * vol_avg_20[i]):          # volume spike
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Camarilla S1 + 1d downtrend + volume confirmation
            elif (close[i] < camarilla_s1_aligned[i] and   # breakdown below S1
                  close[i] < ema_34_1d_aligned[i] and      # price below 1d EMA (downtrend)
                  volume[i] > 1.8 * vol_avg_20[i]):        # volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to opposite Camarilla level or trend reversal
            if position == 1:
                # Exit long: price returns to S1 or trend turns down
                if (close[i] < camarilla_s1_aligned[i] or 
                    close[i] < ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: price returns to R1 or trend turns up
                if (close[i] > camarilla_r1_aligned[i] or 
                    close[i] > ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_1dEMA34_VolumeConfirm"
timeframe = "12h"
leverage = 1.0