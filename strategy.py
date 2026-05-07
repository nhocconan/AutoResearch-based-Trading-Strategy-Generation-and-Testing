#!/usr/bin/env python3
name = "1d_TRIX_VolumeSpike_1wTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # TRIX calculation on weekly close (15-period EMA of EMA of EMA, then ROC)
    close_1w = pd.Series(df_1w['close'])
    ema1 = close_1w.ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix_raw = 100 * (ema3.pct_change())
    trix = trix_raw.fillna(0).values
    trix_signal = pd.Series(trix).ewm(span=9, adjust=False, min_periods=9).mean().values
    
    # Weekly EMA(34) for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align TRIX signal and weekly EMA to daily timeframe
    trix_signal_aligned = align_htf_to_ltf(prices, df_1w, trix_signal)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume spike detection: 20-period average (20 days)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Wait for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(trix_signal_aligned[i]) or np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: TRIX crosses above signal line with volume and weekly uptrend
            trix_cross_up = trix[i] > trix_signal_aligned[i] and trix[i-1] <= trix_signal_aligned[i-1]
            vol_condition = volume[i] > vol_ma_20[i] * 2.0
            weekly_uptrend = ema_34_1w_aligned[i] > ema_34_1w_aligned[i-1]
            
            if trix_cross_up and vol_condition and weekly_uptrend:
                signals[i] = 0.30
                position = 1
            # Short: TRIX crosses below signal line with volume and weekly downtrend
            elif trix[i] < trix_signal_aligned[i] and trix[i-1] >= trix_signal_aligned[i-1] and \
                 vol_condition and not weekly_uptrend:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit: TRIX crosses below signal line or volume drops
            trix_cross_down = trix[i] < trix_signal_aligned[i] and trix[i-1] >= trix_signal_aligned[i-1]
            if trix_cross_down or volume[i] < vol_ma_20[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit: TRIX crosses above signal line or volume drops
            trix_cross_up = trix[i] > trix_signal_aligned[i] and trix[i-1] <= trix_signal_aligned[i-1]
            if trix_cross_up or volume[i] < vol_ma_20[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

# Hypothesis: TRIX momentum with volume confirmation on weekly timeframe
# - TRIX (15,15,15) identifies momentum shifts via triple-smoothed EMA ROC
# - Signal line (9-period EMA of TRIX) provides entry/exit triggers
# - Volume spike (2x 20-day average) confirms institutional participation
# - Weekly EMA(34) filter ensures trades align with higher-timeframe trend
# - Works in bull markets (buy TRIX crosses up in uptrend) and bear markets (sell crosses down in downtrend)
# - Exit on TRIX signal cross or volume weakening to avoid whipsaws
# - Position size 0.30 targets 15-30 trades/year, minimizing fee drag
# - Weekly TRIX + volume + trend filter provides robust edge across market regimes