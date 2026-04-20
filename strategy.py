#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h TRIX with 1d trend filter and volume confirmation.
# TRIX (12-period) filters noise and identifies momentum changes. 
# Long when TRIX crosses above signal line and price above 1d EMA200.
# Short when TRIX crosses below signal line and price below 1d EMA200.
# Volume confirmation ensures institutional participation.
# Designed to work in both bull and bear markets by following higher timeframe trend.
# Target: 20-30 trades per year to minimize fee drag.

name = "4h_TRIX_1dEMA200_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop for EMA200 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === 1d EMA200 for trend direction ===
    close_1d = df_1d['close'].values
    ema_200 = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # === TRIX (12-period) ===
    close = prices['close'].values
    # Triple EMA: EMA(EMA(EMA(close, 12), 12), 12)
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    # TRIX = (ema3 - ema3.shift(1)) / ema3.shift(1) * 100
    trix_raw = (ema3 - ema3.shift(1)) / ema3.shift(1) * 100
    trix = trix_raw.fillna(0).values
    # Signal line: 9-period EMA of TRIX
    trix_signal = pd.Series(trix).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # === 4h Volume confirmation ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after warmup
        # Get values
        trix_val = trix[i]
        trix_signal_val = trix_signal[i]
        ema_val = ema_200_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(trix_val) or np.isnan(trix_signal_val) or 
            np.isnan(ema_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: TRIX crosses above signal line and price above 1d EMA200
            if trix_val > trix_signal_val and trix[i-1] <= trix_signal[i-1] and close[i] > ema_val and vol_ratio_val > 1.5:
                signals[i] = 0.25
                position = 1
            # Short: TRIX crosses below signal line and price below 1d EMA200
            elif trix_val < trix_signal_val and trix[i-1] >= trix_signal[i-1] and close[i] < ema_val and vol_ratio_val > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: TRIX crosses below signal line
            if trix_val < trix_signal_val and trix[i-1] >= trix_signal[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: TRIX crosses above signal line
            if trix_val > trix_signal_val and trix[i-1] <= trix_signal[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals