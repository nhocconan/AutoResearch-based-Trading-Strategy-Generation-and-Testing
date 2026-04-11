#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_cci_reversal_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily CCI(20)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tp_1d = (high_1d + low_1d + close_1d) / 3
    sma_tp = pd.Series(tp_1d).rolling(window=20, min_periods=20).mean()
    mad = pd.Series(tp_1d).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci_1d = (tp_1d - sma_tp.values) / (0.015 * mad.values)
    
    # Align daily CCI to 6h timeframe
    cci_1d_aligned = align_htf_to_ltf(prices, df_1d, cci_1d)
    
    # 6h ATR for volatility filter (14 period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 6h volume filter: volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(cci_1d_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation (1.3x average)
        volume_confirmed = volume_current > 1.3 * vol_ma
        
        # CCI reversal signals: long when CCI < -100, short when CCI > +100
        cci_long_signal = cci_1d_aligned[i] < -100
        cci_short_signal = cci_1d_aligned[i] > 100
        
        # Exit when CCI returns to neutral zone (-50 to 50)
        exit_long = position == 1 and cci_1d_aligned[i] > -50
        exit_short = position == -1 and cci_1d_aligned[i] < 50
        
        # Trading logic
        if cci_long_signal and volume_confirmed and position != 1:
            position = 1
            signals[i] = 0.25
        elif cci_short_signal and volume_confirmed and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Daily CCI reversal strategy for 6h timeframe with volume confirmation.
# Enters long when daily CCI < -100 (oversold) with volume >1.3x average.
# Enters short when daily CCI > +100 (overbought) with volume >1.3x average.
# Exits when CCI returns to neutral zone (-50 to 50) to capture mean reversion.
# CCI is effective in ranging markets which dominate BTC/ETH price action.
# Volume confirmation filters out low-conviction signals.
# Target: 60-120 total trades over 4 years (15-30/year) to minimize fee drag.
# Works in both bull and bear markets as it captures reversals from extremes.