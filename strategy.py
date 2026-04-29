#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h TRIX(9) crossover with 1d EMA34 trend filter and volume spike confirmation
# TRIX catches momentum reversals with reduced whipsaw vs MACD
# 1d EMA34 ensures alignment with primary trend; volume >1.8x confirms institutional participation
# Discrete sizing (0.25) minimizes fee churn; target 75-200 total trades over 4 years

name = "4h_TRIX9_EMA34_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate TRIX (9-period) - triple smoothed EMA of ROC
    close_series = pd.Series(close)
    roc = close_series.pct_change(periods=1)
    ema1 = roc.ewm(span=9, adjust=False, min_periods=9).mean()
    ema2 = ema1.ewm(span=9, adjust=False, min_periods=9).mean()
    ema3 = ema2.ewm(span=9, adjust=False, min_periods=9).mean()
    trix = (ema3 * 100).values  # scale for readability
    
    # Calculate TRIX signal line (9-period EMA of TRIX)
    trix_series = pd.Series(trix)
    trix_signal = trix_series.ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # Calculate ATR for volatility (14-period)
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr2.iloc[0] = 0
    tr3.iloc[0] = 0
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(36, 20, 14, 34)  # warmup: TRIX needs 36 bars (9*4), vol MA needs 20
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(trix[i]) or np.isnan(trix_signal[i]) or 
            np.isnan(atr[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        curr_trix = trix[i]
        curr_trix_signal = trix_signal[i]
        prev_trix = trix[i-1]
        prev_trix_signal = trix_signal[i-1]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Bullish entry: TRIX crosses above signal line + above 1d EMA34 + volume confirmation
            if (prev_trix <= prev_trix_signal and curr_trix > curr_trix_signal and 
                close[i] > curr_ema_34_1d and curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Bearish entry: TRIX crosses below signal line + below 1d EMA34 + volume confirmation
            elif (prev_trix >= prev_trix_signal and curr_trix < curr_trix_signal and 
                  close[i] < curr_ema_34_1d and curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit: TRIX crosses below signal line
            if prev_trix >= prev_trix_signal and curr_trix < curr_trix_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: TRIX crosses above signal line
            if prev_trix <= prev_trix_signal and curr_trix > curr_trix_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals