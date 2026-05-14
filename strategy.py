#!/usr/bin/env python3
# Hypothesis: 1h EMA21 trend pullback with 4h trend alignment and volume confirmation.
# Long when price retraces to EMA21 in a 4h uptrend with above-average volume.
# Short when price retraces to EMA21 in a 4h downtrend with above-average volume.
# Exit on opposite EMA21 cross or volume drop. Uses 4h for trend direction, 1h for entry timing.
# Designed to work in bull/bear: 4h EMA50 filters trend, EMA21 captures pullbacks, volume confirms participation.
# Target: 60-150 total trades over 4 years = 15-37/year for 1h timeframe.

name = "1h_EMA21_Pullback_4hEMA50_Volume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 1h Indicators (LTF) ---
    # EMA21 for pullback entries
    close_s = pd.Series(close)
    ema_21 = close_s.ewm(span=21, adjust=False, min_periods=21).mean().values
    # Volume confirmation: > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    # --- 4h Indicators (HTF) ---
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    
    # 4h EMA50 trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 4h EMA50 uptrend/downtrend (based on slope)
    ema_50_slope = np.gradient(ema_50_4h_aligned)
    ema_50_uptrend = ema_50_slope > 0
    ema_50_downtrend = ema_50_slope < 0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(21, n):  # Start after EMA21 warmup
        # Skip if missing data
        if (np.isnan(ema_21[i]) or 
            np.isnan(ema_50_uptrend[i]) or
            np.isnan(ema_50_downtrend[i]) or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price at EMA21 in 4h uptrend with volume confirmation
            if (close[i] <= ema_21[i] * 1.005 and  # Within 0.5% above EMA21
                close[i] >= ema_21[i] * 0.995 and  # Within 0.5% below EMA21
                ema_50_uptrend[i] and 
                volume_confirm[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Price at EMA21 in 4h downtrend with volume confirmation
            elif (close[i] <= ema_21[i] * 1.005 and 
                  close[i] >= ema_21[i] * 0.995 and 
                  ema_50_downtrend[i] and 
                  volume_confirm[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below EMA21 or volume drops
            if (close[i] < ema_21[i] * 0.995 or  # Below EMA21
                not volume_confirm[i]):           # Volume drop
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price crosses above EMA21 or volume drops
            if (close[i] > ema_21[i] * 1.005 or  # Above EMA21
                not volume_confirm[i]):          # Volume drop
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals