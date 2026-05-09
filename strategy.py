#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WilliamsVixFix_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Williams Vix Fix and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 22:
        return np.zeros(n)
    
    # Calculate 22-period highest close for WVF (as per Larry Williams)
    highest_close = pd.Series(df_1d['close']).rolling(window=22, min_periods=22).max().values
    
    # Williams Vix Fix: measures volatility as inverse of price range from high
    # WVF = ((highest_close - low) / highest_close) * 100
    wvf_raw = ((highest_close - df_1d['low'].values) / highest_close) * 100
    
    # Calculate 10-period moving average of WVF for signal generation
    wf_ma = pd.Series(wvf_raw).rolling(window=10, min_periods=10).mean().values
    
    # Calculate 34-period EMA on daily close for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align daily indicators to 6h timeframe
    wf_ma_aligned = align_htf_to_ltf(prices, df_1d, wf_ma)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-period volume average for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 22, 20)  # Need 34 for EMA, 22 for WVF, 20 for volume
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(wf_ma_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        wf_value = wf_ma_aligned[i]
        ema_1d = ema_34_1d_aligned[i]
        vol = volume[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Enter long: WVF mean crosses above rising threshold in uptrend with volume
            # Rising threshold: 20 + 0.1 * i (adaptive threshold increases slowly)
            threshold = 20 + 0.1 * (i - start_idx)
            if (wf_value > threshold and 
                wf_ma_aligned[i] > wf_ma_aligned[i-1] and  # WVF rising
                vol > 2.0 * vol_ma_val and                 # Volume spike
                close[i] > ema_1d):                        # Price above daily EMA (uptrend)
                signals[i] = 0.25
                position = 1
            # Enter short: WVF mean crosses below falling threshold in downtrend with volume
            elif (wf_value < (threshold - 10) and 
                  wf_ma_aligned[i] < wf_ma_aligned[i-1] and  # WVF falling
                  vol > 2.0 * vol_ma_val and                 # Volume spike
                  close[i] < ema_1d):                        # Price below daily EMA (downtrend)
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: WVF falls below threshold OR trend reverses
            if (wf_value < threshold or 
                close[i] < ema_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: WVF rises above threshold OR trend reverses
            if (wf_value > (threshold - 10) or 
                close[i] > ema_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals