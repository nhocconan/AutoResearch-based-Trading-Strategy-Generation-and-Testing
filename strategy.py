# 12h_1d_trix_volume_regime
# Hypothesis: 12-hour TRIX momentum with volume confirmation and 1-day chop regime filter. TRIX filters noise in choppy markets while capturing momentum in trending regimes. Works in both bull and bear by adapting to market conditions via chop filter.
# Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag.

name = "12h_1d_trix_volume_regime"
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
    
    # Get daily data for TRIX and chop regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # TRIX calculation (15-period EMA of EMA of EMA of price)
    ema1 = pd.Series(close_1d).ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix_raw = ema3.pct_change() * 100
    trix = trix_raw.values
    
    # Chop regime calculation (14-period)
    def true_range(high, low, close_prev):
        tr1 = high - low
        tr2 = np.abs(high - close_prev)
        tr3 = np.abs(low - close_prev)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    close_prev_1d = np.roll(close_1d, 1)
    tr = true_range(high_1d, low_1d, close_prev_1d)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Chop: sum of true ranges over 14 periods / (max(high) - min(low)) over 14 periods
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_denom = highest_high_14 - lowest_low_14
    chop = np.where(chop_denom != 0, (pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values / chop_denom) * 100, 50)
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.3)
    
    # Align TRIX and chop to 12h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(trix_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: TRIX crosses above 0 with volume confirmation and chop < 61.8 (trending)
        if (i > 0 and trix_aligned[i-1] <= 0 and trix_aligned[i] > 0 and 
            vol_confirm[i] and chop_aligned[i] < 61.8 and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: TRIX crosses below 0 with volume confirmation and chop < 61.8 (trending)
        elif (i > 0 and trix_aligned[i-1] >= 0 and trix_aligned[i] < 0 and 
              vol_confirm[i] and chop_aligned[i] < 61.8 and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: TRIX crosses zero in opposite direction or chop > 61.8 (choppy regime)
        elif position == 1 and (trix_aligned[i] < 0 or chop_aligned[i] > 61.8):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (trix_aligned[i] > 0 or chop_aligned[i] > 61.8):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals