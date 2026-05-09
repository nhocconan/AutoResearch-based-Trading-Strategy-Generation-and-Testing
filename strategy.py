#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Keltner_Reversal_DailyTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    """
    6h Keltner reversal with 1d trend filter and volume confirmation.
    - Long: Close crosses below ATR(10) lower band, volume > 1.5x avg, and price > 1d EMA(34)
    - Short: Close crosses above ATR(10) upper band, volume > 1.5x avg, and price < 1d EMA(34)
    - Exit: Close crosses back through EMA(20) or opposite reversal signal
    - Uses 1d EMA(34) for trend filter
    - Target: 12-30 trades/year on 6h timeframe
    """
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 6h EMA(20) for exit
    close_series = pd.Series(close)
    ema20 = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate ATR(10) for Keltner channels
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first TR is just high-low
    atr_series = pd.Series(tr)
    atr10 = atr_series.ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Calculate Keltner channels (EMA20 ± ATR10*2)
    upper = ema20 + 2 * atr10
    lower = ema20 - 2 * atr10
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(ema20[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(vol_ma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma20[i]
        
        if position == 0:
            # Long: Close crosses below lower band (mean reversion) with volume and trend filter
            if close[i] < lower[i] and close[i-1] >= lower[i-1] and vol_ok and close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close crosses above upper band (mean reversion) with volume and trend filter
            elif close[i] > upper[i] and close[i-1] <= upper[i-1] and vol_ok and close[i] < ema34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Close crosses back above EMA20 or opposite signal
            if close[i] > ema20[i] and close[i-1] <= ema20[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Close crosses back below EMA20 or opposite signal
            if close[i] < ema20[i] and close[i-1] >= ema20[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals