#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot levels (R1/S1) breakout with 1d EMA34 trend filter and volume spike confirmation.
# Uses 1d EMA(34) for trend direction, Camarilla R1/S1 levels for entry triggers, and volume > 1.5x 20-period MA for confirmation.
# Long when price breaks above R1 in uptrend with volume spike; short when price breaks below S1 in downtrend with volume spike.
# Exit on trend reversal (price crosses EMA34) or opposite Camarilla level touch.
# Session filter (08-20 UTC) to avoid low-liquidity hours.
# Target: 20-50 trades/year per symbol (80-200 total) to stay within fee limits.
# Designed to work in both bull and bear markets via trend-following breakouts with volatility confirmation.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1d data for Camarilla pivots and trend filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels (R1, S1) from previous day
    # R1 = close + 1.1*(high - low)/12
    # S1 = close - 1.1*(high - low)/12
    rang = high_1d - low_1d
    r1 = close_1d + 1.1 * rang / 12
    s1 = close_1d - 1.1 * rang / 12
    
    # Align Camarilla levels to 4h timeframe (using previous day's levels)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 1d EMA(34) for trend direction
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike filter (20-period)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if not in session or data not ready
        if not in_session[i] or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1 + uptrend (price > EMA34) + volume spike
            if (close[i] > r1_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + downtrend (price < EMA34) + volume spike
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: trend reversal or opposite Camarilla level touch
            if position == 1:
                if (close[i] < ema_34_1d_aligned[i] or close[i] < s1_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if (close[i] > ema_34_1d_aligned[i] or close[i] > r1_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0