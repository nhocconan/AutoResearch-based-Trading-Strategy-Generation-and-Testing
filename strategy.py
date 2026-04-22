#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray with 1d Trend Filter and Volume Confirmation
# Uses Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) from 6h data
# Long when Bull Power > 0, Bear Power < 0, 1d EMA34 uptrend, and volume spike
# Short when Bull Power < 0, Bear Power > 0, 1d EMA34 downtrend, and volume spike
# Combines trend strength (Elder Ray) with higher timeframe trend and volume confirmation
# Designed for 6h timeframe to target 15-30 trades/year per symbol (60-120 total over 4 years)
# Elder Ray avoids whipsaws by requiring both bull and bear power conditions simultaneously

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for trend filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 6h EMA(13) for Elder Ray calculation
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # High - EMA13
    bear_power = ema_13 - low   # EMA13 - Low
    
    # 1d EMA(34) for higher timeframe trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike filter (20-period on 6h data)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0, Bear Power < 0, 1d uptrend, volume spike
            if (bull_power[i] > 0 and 
                bear_power[i] < 0 and 
                close[i] > ema_34_1d_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bull Power < 0, Bear Power > 0, 1d downtrend, volume spike
            elif (bull_power[i] < 0 and 
                  bear_power[i] > 0 and 
                  close[i] < ema_34_1d_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Elder Ray divergence or trend reversal
            if position == 1:
                # Exit on Bear Power >= 0 (bullish momentum fading) or trend reversal
                if (bear_power[i] >= 0 or 
                    close[i] < ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit on Bull Power <= 0 (bearish momentum fading) or trend reversal
                if (bull_power[i] <= 0 or 
                    close[i] > ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_1dEMA34_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0