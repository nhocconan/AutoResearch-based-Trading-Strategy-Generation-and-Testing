#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Alligator (Jaw/Teeth/Lips) with 1d EMA50 trend filter and volume confirmation.
- Long when Lips > Teeth > Jaw (bullish alignment) AND price > 1d EMA50 AND volume > 1.3 * ATR(14) * close
- Short when Lips < Teeth < Jaw (bearish alignment) AND price < 1d EMA50 AND volume > 1.3 * ATR(14) * close
- Exit when Alligator alignment breaks or price crosses 1d EMA50
- Uses 4h primary timeframe with 1d HTF to target 100-200 trades over 4 years (25-50/year)
- Williams Alligator identifies trending vs ranging markets via jaw/teeth/lips convergence/divergence
- 1d EMA50 ensures alignment with longer-term trend to avoid whipsaws in bear markets
- ATR-scaled volume filter adapts to volatility, reducing false signals during low-volatility periods
- Designed for BTC/ETH with edge in trending markets (Alligator alignment continuation) and range markets (mean reversion at extremes via trend filter)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams Alligator (13,8,5) SMAs using previous values (no look-ahead)
    close_series = pd.Series(close)
    # Jaw: 13-period SMA, shifted by 8 bars
    jaw = close_series.rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth: 8-period SMA, shifted by 5 bars
    teeth = close_series.rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips: 5-period SMA, shifted by 3 bars
    lips = close_series.rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Get 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) for dynamic volume threshold
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr2.iloc[0] = np.nan
    tr3.iloc[0] = np.nan
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Dynamic volume threshold: volume > 1.3 * ATR * close (volatility-adjusted)
    vol_threshold = 1.3 * atr * close
    volume_confirm = volume > vol_threshold
    
    # Alligator alignment signals
    bullish_alignment = (lips > teeth) & (teeth > jaw)
    bearish_alignment = (lips < teeth) & (teeth < jaw)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13+8, 8+5, 5+3, 50, 14) + 1  # ~26
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: bullish Alligator alignment, price above 1d EMA50, volume confirmation
            if bullish_alignment[i] and close[i] > ema_50_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish Alligator alignment, price below 1d EMA50, volume confirmation
            elif bearish_alignment[i] and close[i] < ema_50_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Alligator alignment breaks OR price crosses below 1d EMA50
            if not bullish_alignment[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator alignment breaks OR price crosses above 1d EMA50
            if not bearish_alignment[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsAlligator_1dEMA50_ATRVolConfirm_v1"
timeframe = "4h"
leverage = 1.0