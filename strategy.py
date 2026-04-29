#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h TRIX (15,9) with 1w EMA50 trend filter and volume confirmation (>1.5x 30-period average)
# TRIX filters noise and shows momentum; long when TRIX rising and above zero, short when falling and below zero.
# 1w EMA50 ensures we trade only with the higher timeframe trend to avoid whipsaws.
# Volume confirmation filters for institutional participation; discrete sizing (0.25) minimizes fee churn.
# Effective in both bull and bear markets: catches strong trends when TRIX accelerates, avoids chop when TRIX flattens.
# Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe.

name = "12h_TRIX_1wEMA50_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # TRIX on 12h timeframe: EMA(EMA(EMA(close,15),15),15) then ROC(9)
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix = 100 * (ema3.pct_change(periods=9))
    trix_values = trix.values
    trix_prev = trix_values.copy()
    trix_prev[1:] = trix_values[:-1]  # previous bar
    trix_prev[0] = np.nan
    
    # Calculate 30-period average volume for confirmation (on 12h timeframe)
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 15*3+9, 30)  # 1w EMA50, TRIX warmup, volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(trix_values[i]) or np.isnan(trix_prev[i]) or 
            np.isnan(vol_ma_30[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_ema_1w = ema_50_1w_aligned[i]
        curr_trix = trix_values[i]
        prev_trix = trix_prev[i]
        curr_vol_ma = vol_ma_30[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 1.5x 30-period average
        vol_confirm = curr_volume > 1.5 * curr_vol_ma
        
        # TRIX momentum conditions
        # Long: TRIX rising (current > previous) AND above zero
        # Short: TRIX falling (current < previous) AND below zero
        trix_long = curr_trix > prev_trix and curr_trix > 0
        trix_short = curr_trix < prev_trix and curr_trix < 0
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: TRIX turns down OR trend turns bearish (price below 1w EMA50)
            if not trix_long or curr_close < curr_ema_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: TRIX turns up OR trend turns bullish (price above 1w EMA50)
            if not trix_short or curr_close > curr_ema_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: TRIX rising AND above zero AND above 1w EMA50 AND volume confirmation
            if (trix_long and 
                curr_trix > 0 and 
                curr_close > curr_ema_1w and 
                vol_confirm):
                signals[i] = 0.25
                position = 1
            # Short entry: TRIX falling AND below zero AND below 1w EMA50 AND volume confirmation
            elif (trix_short and 
                  curr_trix < 0 and 
                  curr_close < curr_ema_1w and 
                  vol_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals