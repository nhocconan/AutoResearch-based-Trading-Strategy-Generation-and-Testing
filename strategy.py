# 1h_TRIX_Momentum_4hTrend_Volume
# Hypothesis: TRIX (triple EMA) momentum on 1h captures short-term trends, while 4h EMA50 provides trend filter and volume confirms breakouts.
# Works in bull (momentum continuations) and bear (mean reversion at extremes) with tight entries to avoid overtrading.
# Target: 60-150 total trades over 4 years = 15-37/year for 1h.

name = "1h_TRIX_Momentum_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # 4h EMA50 trend
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_4h_up = close_4h > ema50_4h
    trend_4h_down = close_4h < ema50_4h
    
    # Align 4h trend to 1h
    trend_4h_up_aligned = align_htf_to_ltf(prices, df_4h, trend_4h_up.astype(float))
    trend_4h_down_aligned = align_htf_to_ltf(prices, df_4h, trend_4h_down.astype(float))
    
    # TRIX on 1h: triple EMA of ROC
    # TRIX = EMA(EMA(EMA(ROC, 12), 12), 12)
    close_series = pd.Series(close)
    roc = close_series.pct_change(periods=12)  # Rate of change over 12 periods
    ema1 = roc.ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix = ema3.values
    
    # Volume spike: current > 2.0 * 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(trend_4h_up_aligned[i]) or np.isnan(trend_4h_down_aligned[i]) or
            np.isnan(trix[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_spike = vol_ratio > 2.0
        
        if position == 0:
            # Long: TRIX crosses above zero with 4h uptrend and volume spike
            if (trix[i] > 0 and trix[i-1] <= 0 and 
                trend_4h_up_aligned[i] > 0.5 and volume_spike):
                signals[i] = 0.20
                position = 1
            # Short: TRIX crosses below zero with 4h downtrend and volume spike
            elif (trix[i] < 0 and trix[i-1] >= 0 and 
                  trend_4h_down_aligned[i] > 0.5 and volume_spike):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit: TRIX crosses below zero or trend fails
            if (trix[i] < 0 and trix[i-1] >= 0) or \
               trend_4h_up_aligned[i] < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit: TRIX crosses above zero or trend fails
            if (trix[i] > 0 and trix[i-1] <= 0) or \
               trend_4h_down_aligned[i] < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals