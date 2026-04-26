#!/usr/bin/env python3
"""
4h_TRIX_ZeroCross_VolumeSpike_12hEMA50_Trend_v1
Hypothesis: TRIX (15-period) zero-cross with volume spike confirmation (>2.0x median) and 12h EMA50 trend filter. TRIX captures momentum reversals with less whipsaw than MACD. Volume spike ensures institutional participation. 12h EMA50 provides multi-timeframe trend alignment to avoid counter-trend trades. Designed for BTC/ETH with tight entry conditions (~25-40 trades/year) to minimize fee drag. Works in bull/bear by only trading with 12h trend direction.
"""

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
    
    # Get 12h data for HTF trend (EMA50)
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA(50) for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # TRIX (15-period): Triple EMA of ROC, then ROC of that
    # ROC(1) = (close/prev_close - 1) * 100
    roc = np.zeros_like(close)
    roc[1:] = (close[1:] / close[:-1] - 1) * 100
    roc[0] = 0
    
    # Triple EMA of ROC
    ema1 = pd.Series(roc).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    
    # TRIX = ROC of triple EMA
    trix = np.zeros_like(ema3)
    trix[1:] = (ema3[1:] / ema3[:-1] - 1) * 100
    trix[0] = 0
    
    # Volume spike filter: volume > 2.0x median volume (30-period)
    vol_median = pd.Series(volume).rolling(window=30, min_periods=30).median().values
    
    # ATR(14) for volatility-based stops
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align HTF indicators to 4h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup: max of TRIX (need 45 for triple EMA), volume median (30), ATR (14)
    start_idx = max(45, 30, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(trix[i]) or
            np.isnan(vol_median[i]) or
            np.isnan(atr[i])):
            # Hold current position
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_50_12h_val = ema_50_12h_aligned[i]
        close_val = close[i]
        trix_val = trix[i]
        volume_val = volume[i]
        vol_median_val = vol_median[i]
        atr_val = atr[i]
        
        # Trend filter: price > EMA50 (uptrend) or < EMA50 (downtrend)
        uptrend = close_val > ema_50_12h_val
        downtrend = close_val < ema_50_12h_val
        
        # Volume spike filter: only trade in high-volume environments
        volume_spike = volume_val > 2.0 * vol_median_val
        
        # TRIX signals: zero-cross with confirmation
        trix_cross_up = trix_val > 0 and trix[i-1] <= 0
        trix_cross_down = trix_val < 0 and trix[i-1] >= 0
        
        if position == 0:
            # Long: TRIX crosses above zero with volume spike, and uptrend
            long_signal = trix_cross_up and volume_spike and uptrend
            
            # Short: TRIX crosses below zero with volume spike, and downtrend
            short_signal = trix_cross_down and volume_spike and downtrend
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                highest_since_entry = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                lowest_since_entry = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            highest_since_entry = max(highest_since_entry, high_val)
            # ATR trailing stop (2.0x for responsiveness)
            if close_val < highest_since_entry - 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            lowest_since_entry = min(lowest_since_entry, low_val)
            # ATR trailing stop (2.0x for responsiveness)
            if close_val > lowest_since_entry + 2.0 * atr_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_TRIX_ZeroCross_VolumeSpike_12hEMA50_Trend_v1"
timeframe = "4h"
leverage = 1.0