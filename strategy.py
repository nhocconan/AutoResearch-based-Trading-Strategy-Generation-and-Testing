# 4h_RVOL_Trend_Pullback_Breakout
# Hypothesis: Combines 1d volatility regime (RVOL > 1.5) with 4h pullback to EMA20 in direction of 1d trend. 
# Enters on breakout of pullback high/low with volume confirmation. Works in both bull/bear via trend filter.
# Target: 20-40 trades/year to minimize fee drag on 4h timeframe.

name = "4h_RVOL_Trend_Pullback_Breakout"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1d_up = close_1d > ema50_1d
    trend_1d_down = close_1d < ema50_1d
    
    # Align 1d trend to 4h
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up.astype(float))
    trend_1d_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down.astype(float))
    
    # 1d RVOL (volume / 20-day average)
    vol_20 = np.full(len(df_1d), np.nan)
    vol_sum = 0
    for i in range(len(df_1d)):
        vol_sum += df_1d['volume'].iloc[i]
        if i >= 20:
            vol_sum -= df_1d['volume'].iloc[i-20]
        if i >= 19:
            vol_20[i] = vol_sum / 20
    rvol = df_1d['volume'].values / vol_20
    vol_regime = rvol > 1.5  # High volatility regime
    
    # Align volatility regime to 4h
    vol_regime_aligned = align_htf_to_ltf(prices, df_1d, vol_regime.astype(float))
    
    # 4h EMA20 for pullback
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume confirmation (1.5x 20-period average)
    vol_ma = np.full(n, np.nan)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma[i] = vol_sum / 20
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(trend_1d_up_aligned[i]) or np.isnan(trend_1d_down_aligned[i]) or
            np.isnan(vol_regime_aligned[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: pullback to EMA20 in uptrend, then break above pullback high
            if (trend_1d_up_aligned[i] > 0.5 and
                vol_regime_aligned[i] > 0.5 and
                low[i] <= ema20[i] and  # Pulled back to EMA20
                high[i] > ema20[i] and  # Closed above EMA20
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: pullback to EMA20 in downtrend, then break below pullback low
            elif (trend_1d_down_aligned[i] > 0.5 and
                  vol_regime_aligned[i] > 0.5 and
                  high[i] >= ema20[i] and  # Pulled back to EMA20
                  low[i] < ema20[i] and    # Closed below EMA20
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: trend reversal or volatility drops
            if (trend_1d_up_aligned[i] < 0.5 or
                vol_regime_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: trend reversal or volatility drops
            if (trend_1d_down_aligned[i] < 0.5 or
                vol_regime_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals