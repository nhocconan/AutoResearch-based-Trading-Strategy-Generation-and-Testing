#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with 1d trend filter and volume confirmation.
# Uses daily Camarilla pivot levels (R1, S1) for breakout entries.
# Long when price breaks above R1 with volume confirmation and daily uptrend.
# Short when price breaks below S1 with volume confirmation and daily downtrend.
# Uses daily EMA(34) as higher timeframe trend filter to avoid counter-trend trades.
# Designed for 12h timeframe to target 12-37 trades/year per symbol.
# Works in bull/bear via multi-timeframe trend alignment and volume confirmation.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Camarilla pivots and trend filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each day
    # Pivot = (High + Low + Close) / 3
    # R1 = Close + (High - Low) * 1.1 / 12
    # S1 = Close - (High - Low) * 1.1 / 12
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = close_1d + (high_1d - low_1d) * 1.1 / 12.0
    s1_1d = close_1d - (high_1d - low_1d) * 1.1 / 12.0
    
    # Align Camarilla levels to 12h timeframe (wait for daily close)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Calculate daily EMA(34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike filter (24-period for 12h timeframe)
    vol_ma24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_spike = volume > 2.0 * vol_ma24  # Higher threshold for fewer trades
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1 + daily uptrend + volume spike
            if (close[i] > r1_aligned[i] and 
                close[i] > ema_34_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + daily downtrend + volume spike
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema_34_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit on price below pivot or trend reversal
                if (close[i] < pivot_aligned[i] or close[i] < ema_34_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit on price above pivot or trend reversal
                if (close[i] > pivot_aligned[i] or close[i] > ema_34_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1dEMA34_Trend_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0