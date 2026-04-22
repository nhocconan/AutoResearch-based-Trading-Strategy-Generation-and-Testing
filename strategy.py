# 12h_TRIX_VolumeSpike_1dTrend
# Hypothesis: TRIX (12-period) on 12h timeframe with 1d EMA34 trend filter and volume spike confirmation
# TRIX is a momentum oscillator that filters out insignificant price movements, effective in both trending and ranging markets
# Long when TRIX crosses above zero with 1d uptrend and volume spike
# Short when TRIX crosses below zero with 1d downtrend and volume spike
# Designed for 12h timeframe to target 15-30 trades/year per symbol with strong trend filtering to reduce whipsaws
# Volume spike ensures momentum confirmation, reducing false signals
# TRIX zero-line cross provides clear entry/exit signals with inherent smoothing

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
    
    # Load 1d data for trend filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate TRIX (12-period triple EMA) on 12h data
    # TRIX = EMA(EMA(EMA(close, 12), 12), 12) - 1-period percent change
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    
    # Calculate 1-period percent change of triple EMA
    trix_raw = np.zeros_like(ema3)
    trix_raw[1:] = (ema3[1:] - ema3[:-1]) / ema3[:-1] * 100
    trix_raw[0] = 0.0
    
    # 1d EMA(34) for higher timeframe trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike filter (20-period on 12h data)
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
            # Long: TRIX crosses above zero + 1d uptrend + volume spike
            if (trix_raw[i] > 0 and trix_raw[i-1] <= 0 and 
                close[i] > ema_34_1d_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below zero + 1d downtrend + volume spike
            elif (trix_raw[i] < 0 and trix_raw[i-1] >= 0 and 
                  close[i] < ema_34_1d_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: TRIX crosses zero in opposite direction or trend reversal
            if position == 1:
                # Exit on TRIX cross below zero or trend reversal
                if (trix_raw[i] < 0 and trix_raw[i-1] >= 0) or \
                   (close[i] < ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit on TRIX cross above zero or trend reversal
                if (trix_raw[i] > 0 and trix_raw[i-1] <= 0) or \
                   (close[i] > ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_TRIX_VolumeSpike_1dTrend"
timeframe = "12h"
leverage = 1.0