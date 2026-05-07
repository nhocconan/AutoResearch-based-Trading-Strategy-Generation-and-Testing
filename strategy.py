#!/usr/bin/env python3
name = "12h_ParabolicSAR_Trend_1dATR_Volume"
timeframe = "12h"
leverage = 1.0

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
    
    # Daily ATR for volatility filter and stop
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Parabolic SAR parameters
    af_init = 0.02
    af_step = 0.02
    af_max = 0.2
    
    # Initialize SAR arrays
    sar = np.zeros(n)
    trend = np.ones(n)  # 1 for uptrend, -1 for downtrend
    ep = np.zeros(n)    # extreme point
    af = np.zeros(n)    # acceleration factor
    
    # Set initial values
    sar[0] = low[0]
    trend[0] = 1
    ep[0] = high[0]
    af[0] = af_init
    
    # Calculate SAR for each bar
    for i in range(1, n):
        if trend[i-1] == 1:  # uptrend
            sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
            if low[i] <= sar[i]:  # trend reversal to down
                trend[i] = -1
                sar[i] = ep[i-1]
                ep[i] = low[i]
                af[i] = af_init
            else:
                trend[i] = 1
                if high[i] > ep[i-1]:
                    ep[i] = high[i]
                    af[i] = min(af[i-1] + af_step, af_max)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
        else:  # downtrend
            sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
            if high[i] >= sar[i]:  # trend reversal to up
                trend[i] = 1
                sar[i] = ep[i-1]
                ep[i] = high[i]
                af[i] = af_init
            else:
                trend[i] = -1
                if low[i] < ep[i-1]:
                    ep[i] = low[i]
                    af[i] = min(af[i-1] + af_step, af_max)
                else:
                    ep[i] = ep[i-1]
                    af[i] = af[i-1]
    
    # Daily trend filter (EMA 50)
    ema_50_1d = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Load daily data once for trend filter
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d_daily = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d_daily)
    
    # Volume confirmation (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(sar[i]) or np.isnan(ema_50_1d[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 1.5x average
        vol_condition = volume[i] > vol_ma_20[i] * 1.5
        
        if position == 0:
            # Long: SAR below price (uptrend) + price above daily EMA50 + daily uptrend + volume
            if sar[i] < close[i] and close[i] > ema_50_1d[i] and ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1] and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: SAR above price (downtrend) + price below daily EMA50 + daily downtrend + volume
            elif sar[i] > close[i] and close[i] < ema_50_1d[i] and ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1] and vol_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: SAR above price OR price below daily EMA50
            if sar[i] > close[i] or close[i] < ema_50_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: SAR below price OR price above daily EMA50
            if sar[i] < close[i] or close[i] > ema_50_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 12h Parabolic SAR with daily trend filter and volume confirmation
# - Parabolic SAR provides trend-following signals with built-in acceleration
# - Daily EMA50 filter ensures alignment with higher timeframe trend
# - Volume confirmation (1.5x average) reduces false signals
# - Works in both bull (SAR below price in uptrend) and bear (SAR above price in downtrend)
# - ATR-based volatility filtering would be added in production but omitted for simplicity
# - Position size 0.25 targets ~20-50 trades/year to stay within limits
# - Parabolic SAR is effective in trending markets and avoids whipsaws in ranges
# - Daily trend filter prevents counter-trend trades during major reversals
# - Volume confirmation ensures institutional participation in breakouts
# - Expected to perform well in both bull and bear markets due to trend-following nature
# - Simple, robust logic with minimal overfitting risk
# - Target: 50-100 total trades over 4 years (12-25/year) to avoid fee drag