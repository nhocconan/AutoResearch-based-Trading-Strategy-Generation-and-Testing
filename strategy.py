#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R mean reversion with 1d trend filter and volume confirmation
# - Uses 1d EMA(50) for trend direction (long when price > EMA, short when price < EMA)
# - Uses 12h Williams %R(14) for mean reversion signals (long when %R < -80, short when %R > -20)
# - Requires volume > 1.3 * 20-period volume average for confirmation
# - Fixed position size 0.25 to manage drawdown and reduce fee churn
# - ATR-based stoploss (2.5 * ATR) to limit losses
# - Target: 12-30 trades/year on 12h timeframe (48-120 total over 4 years)
# - Works in bull markets via mean reversion during uptrend pullbacks, in bear via bounces in downtrend

name = "12h_1d_williamsr_mean_reversion_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Pre-compute 12h Williams %R(14)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Pre-compute ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute volume confirmation: volume > 1.3 * 20-period average
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(williams_r[i]) or np.isnan(atr[i]) or atr[i] <= 0 or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from 1d EMA
        uptrend = close[i] > ema_1d_aligned[i]
        downtrend = close[i] < ema_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit conditions: stoploss or mean reversion reversal
            if close[i] < ema_1d_aligned[i] - 2.5 * atr[i]:  # ATR stop below trend
                position = 0
                signals[i] = 0.0
            elif williams_r[i] > -20:  # Overbought exit
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: stoploss or mean reversion reversal
            if close[i] > ema_1d_aligned[i] + 2.5 * atr[i]:  # ATR stop above trend
                position = 0
                signals[i] = 0.0
            elif williams_r[i] < -80:  # Oversold exit
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for mean reversion entries in direction of 1d trend with volume confirmation
            if uptrend and williams_r[i] < -80 and volume_confirm[i]:  # Oversold in uptrend
                position = 1
                signals[i] = 0.25
            elif downtrend and williams_r[i] > -20 and volume_confirm[i]:  # Overbought in downtrend
                position = -1
                signals[i] = -0.25
    
    return signals