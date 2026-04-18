#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h TRIX (12,9) with 1d volume confirmation and volatility filter.
# TRIX measures momentum as % rate of change of triple-smoothed EMA.
# Long when TRIX > 0 and 1d volume > 1.5x 20-day average; short when TRIX < 0 and volume condition.
# 1d volume filter ensures institutional participation. Volatility filter (ATR ratio) avoids chop.
# Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag.
name = "4h_TRIX_1dVolumeVolFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate TRIX (12,9): triple EMA then % ROC
    close_s = pd.Series(close)
    ema1 = close_s.ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix = ema3.pct_change(periods=1) * 100
    trix_values = trix.values
    
    # Calculate 1d volume average
    vol_1d = pd.Series(df_1d['volume'].values)
    vol_ma_20_1d = vol_1d.rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # ATR ratio: current ATR / 50-period average ATR (to detect low volatility)
    atr_ma_50 = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr / atr_ma_50  # < 1 = low volatility, > 1 = high volatility
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(trix_values[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or
            np.isnan(atr_ratio[i]) or np.isnan(atr_ma_50[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: 1d volume > 1.5x 20-day average
        volume_filter = df_1d['volume'].values[min(i // 24, len(df_1d)-1)] > (1.5 * vol_ma_20_1d_aligned[i]) if i >= 24 else False
        
        # Volatility filter: avoid extremely low volatility (chop)
        vol_filter = atr_ratio[i] > 0.8
        
        if position == 0:
            # Long: TRIX positive AND volume filter AND volatility filter
            if trix_values[i] > 0 and volume_filter and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: TRIX negative AND volume filter AND volatility filter
            elif trix_values[i] < 0 and volume_filter and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: TRIX turns negative OR volume filter fails
            if trix_values[i] <= 0 or not volume_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: TRIX turns positive OR volume filter fails
            if trix_values[i] >= 0 or not volume_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals