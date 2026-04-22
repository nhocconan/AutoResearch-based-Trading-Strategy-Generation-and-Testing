#!/usr/bin/env python3
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
    
    # Load weekly data for trend (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Weekly EMA34 for trend
    weekly_close = df_1w['close'].values
    ema34_1w = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Daily ATR for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Daily volume average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if np.isnan(ema34_1w_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_avg_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above weekly EMA34 + volume spike + volatility expansion
            if (close[i] > ema34_1w_aligned[i] and 
                volume[i] > 2.0 * vol_avg_20[i] and
                atr[i] > 1.5 * atr[i-1] if i > 0 else True):
                signals[i] = 0.30
                position = 1
            # Short: Price below weekly EMA34 + volume spike + volatility expansion
            elif (close[i] < ema34_1w_aligned[i] and 
                  volume[i] > 2.0 * vol_avg_20[i] and
                  atr[i] > 1.5 * atr[i-1] if i > 0 else True):
                signals[i] = -0.30
                position = -1
        else:
            # Exit: Price crosses back to weekly EMA34
            if position == 1:
                if close[i] < ema34_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.30
            else:  # position == -1
                if close[i] > ema34_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.30
    
    return signals

name = "1D_WeeklyEMA34_Trend_VolumeVolatility"
timeframe = "1d"
leverage = 1.0