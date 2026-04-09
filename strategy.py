#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with weekly trend filter and volume confirmation
# - Uses 1w EMA(21) for trend direction (long when close > EMA, short when close < EMA)
# - Uses 6h Williams %R(14) for mean reversion entries (long when %R < -80, short when %R > -20)
# - Requires volume > 1.5x 20-period average for confirmation
# - Fixed position size 0.25 to manage drawdown
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Works in bull markets via pullbacks in uptrends, in bear via bounces in downtrends

name = "6h_1d_1w_williamsr_meanrev_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 30 or len(df_1d) < 50:
        return np.zeros(n)
    
    # 1w EMA(21) for trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # 1d ATR(14) for dynamic thresholds
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Pre-compute 6h Williams %R(14)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    
    # Pre-compute 6h volume ratio (current vs 20-period average)
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(williams_r[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr_1d_aligned[i]) or atr_1d_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from 1w EMA
        uptrend = close[i] > ema_1w_aligned[i]
        downtrend = close[i] < ema_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit conditions: mean reversion or trend change
            if williams_r[i] > -20:  # Williams %R overbought exit
                position = 0
                signals[i] = 0.0
            elif not uptrend:  # Trend changed to downtrend or sideways
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: mean reversion or trend change
            if williams_r[i] < -80:  # Williams %R oversold exit
                position = 0
                signals[i] = 0.0
            elif not downtrend:  # Trend changed to uptrend or sideways
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for mean reversion entries in direction of 1w trend
            if uptrend and williams_r[i] < -80 and vol_ratio[i] > 1.5:  # Oversold in uptrend with volume
                position = 1
                signals[i] = 0.25
            elif downtrend and williams_r[i] > -20 and vol_ratio[i] > 1.5:  # Overbought in downtrend with volume
                position = -1
                signals[i] = -0.25
    
    return signals