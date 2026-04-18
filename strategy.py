#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Weekly EMA(21) with volume breakout and volatility filter.
# Long when price crosses above weekly EMA(21) with volume > 2x 10-day average and ATR > 0.
# Short when price crosses below weekly EMA(21) with same conditions.
# Exit when price crosses back over weekly EMA(21).
# Weekly trend filter reduces whipsaw, volume surge adds conviction, ATR filter avoids low-volatility noise.
# Designed for ~10-20 trades/year per symbol (~40-80 total over 4 years).
name = "1d_WeeklyEMA21_VolumeBreakout_ATR_Filter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # EMA(21) on weekly close
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # ATR(14) on weekly for volatility filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - close_1w)
    tr3 = np.abs(low_1w - close_1w)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Volume filter: current volume > 2.0 * 10-period average (10-day average on daily)
    vol_ma_10 = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    volume_filter = volume > (2.0 * vol_ma_10)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_21_1w_aligned[i]) or np.isnan(atr_1w_aligned[i]) or
            np.isnan(vol_ma_10[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_val = ema_21_1w_aligned[i]
        atr_val = atr_1w_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Long: price crosses above weekly EMA with volume surge and volatility
            if close_val > ema_val and vol_filter and atr_val > 0:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below weekly EMA with volume surge and volatility
            elif close_val < ema_val and vol_filter and atr_val > 0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses back below weekly EMA
            if close_val < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back above weekly EMA
            if close_val > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals