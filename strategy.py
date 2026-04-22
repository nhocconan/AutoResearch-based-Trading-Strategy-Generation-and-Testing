# 4H_Camarilla_Pivot_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Camarilla pivot levels (R1, S1) from daily timeframe act as intraday support/resistance.
# Breakout above R1 with 1-day trend confirmation (price > EMA34) and volume spike signals bullish momentum.
# Breakdown below S1 with 1-day trend confirmation (price < EMA34) and volume spike signals bearish momentum.
# Works in both bull and bear markets by following institutional price action at key levels with volume confirmation.
# Uses 4h timeframe for optimal trade frequency (target: 20-50 trades/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_hlf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1-day data for Camarilla pivot, trend, and volume filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for each day
    # R1 = Close + 1.1*(High-Low)/12
    # S1 = Close - 1.1*(High-Low)/12
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    camarilla_r1 = close_1d + 1.1 * (high_1d - low_1d) / 12
    camarilla_s1 = close_1d - 1.1 * (high_1d - low_1d) / 12
    
    # Align Camarilla levels to 4h timeframe
    r1_4h = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_4h = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # 1-day EMA34 for trend confirmation
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1-day volume average for spike detection (20-period)
    volume_1d = df_1d['volume'].values
    avg_vol_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    avg_vol_20_4h = align_htf_to_ltf(prices, df_1d, avg_vol_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or 
            np.isnan(ema_34_4h[i]) or np.isnan(avg_vol_20_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Breakout above R1 with uptrend and volume spike
            if (close[i] > r1_4h[i] and 
                close[i] > ema_34_4h[i] and 
                volume[i] > 2.0 * avg_vol_20_4h[i]):
                signals[i] = 0.25
                position = 1
            # Short: Breakdown below S1 with downtrend and volume spike
            elif (close[i] < s1_4h[i] and 
                  close[i] < ema_34_4h[i] and 
                  volume[i] > 2.0 * avg_vol_20_4h[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price returns to opposite Camarilla level
            exit_signal = False
            
            if position == 1:
                # Exit long: Price falls back to S1
                if close[i] < s1_4h[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price rises back to R1
                if close[i] > r1_4h[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_Pivot_R1_S1_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0