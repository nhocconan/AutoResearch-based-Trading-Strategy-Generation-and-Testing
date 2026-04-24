#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator + 1d EMA50 trend filter + volume confirmation.
- Long when Alligator jaws (SMMA13) > teeth (SMMA8) > lips (SMMA5) AND close > 1d EMA50 AND volume > 1.5 * ATR(14) * close
- Short when Alligator jaws < teeth < lips AND close < 1d EMA50 AND volume > 1.5 * ATR(14) * close
- Exit when Alligator alignment breaks (jaws not > teeth > lips for long, or jaws not < teeth < lips for short)
- Uses 12h primary timeframe with 1d HTF to target 50-150 trades over 4 years (12-37/year)
- Williams Alligator identifies trend strength via SMMA alignment, reducing whipsaws in choppy markets
- 1d EMA50 ensures alignment with longer-term trend to avoid counter-trend entries
- ATR-scaled volume filter confirms breakouts with institutional participation
- Designed for BTC/ETH with edge in trending markets (Alligator alignment) and bear markets (avoiding false signals via trend filter)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(source, period):
    """Smoothed Moving Average (SMMA) - also called RMA or Wilder's MA"""
    if len(source) < period:
        return np.full_like(source, np.nan, dtype=np.float64)
    result = np.empty_like(source, dtype=np.float64)
    result[:] = np.nan
    # First value is simple SMA
    result[period-1] = np.mean(source[:period])
    # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT) / period
    for i in range(period, len(source)):
        result[i] = (result[i-1] * (period-1) + source[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams Alligator components (SMMA) - using previous close to avoid look-ahead
    prev_close = np.roll(close, 1)
    prev_close[0] = np.nan
    
    # SMMA5 (lips), SMMA8 (teeth), SMMA13 (jaws)
    lips = smma(prev_close, 5)
    teeth = smma(prev_close, 8)
    jaws = smma(prev_close, 13)
    
    # Get 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 12h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) for dynamic volume threshold
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr2.iloc[0] = np.nan
    tr3.iloc[0] = np.nan
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Dynamic volume threshold: volume > 1.5 * ATR * close (volatility-adjusted)
    vol_threshold = 1.5 * atr * close
    volume_confirm = volume > vol_threshold
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(13, 50, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator alignment conditions
        bullish_alignment = jaws[i] > teeth[i] and teeth[i] > lips[i]
        bearish_alignment = jaws[i] < teeth[i] and teeth[i] < lips[i]
        
        if position == 0:
            # Long: bullish Alligator alignment, trend up (close > EMA50), volume confirmation
            if bullish_alignment and close[i] > ema_50_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish Alligator alignment, trend down (close < EMA50), volume confirmation
            elif bearish_alignment and close[i] < ema_50_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Alligator bullish alignment breaks
            if not (jaws[i] > teeth[i] and teeth[i] > lips[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator bearish alignment breaks
            if not (jaws[i] < teeth[i] and teeth[i] < lips[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dEMA50_ATRVolConfirm_v1"
timeframe = "12h"
leverage = 1.0