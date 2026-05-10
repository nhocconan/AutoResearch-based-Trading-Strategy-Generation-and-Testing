# State your hypothesis in a comment at the top (strategy type, timeframe, why it should work in BOTH bull AND bear)
# Hypothesis: In trending markets, price often respects the 50-period EMA as dynamic support/resistance.
# We combine this with 1-day ATR-based volatility breakouts to capture momentum moves.
# Long when: price > 50 EMA AND breaks above (previous close + 0.5 * ATR)
# Short when: price < 50 EMA AND breaks below (previous close - 0.5 * ATR)
# Uses 1-day ATR for volatility scaling to adapt to changing market conditions.
# Works in bull markets (follows uptrends with breakouts) and bear markets (follows downtrends with breakdowns).
# Volume confirmation filters low-conviction moves.
# Target: 20-40 trades/year to stay well under the 400-trade limit and minimize fee drag.

name = "4h_EMA50_ATRBreakout_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 14-period ATR on daily timeframe
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    atr_14 = np.zeros_like(tr)
    for i in range(1, len(tr)):
        atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14  # Wilder's smoothing
    
    # Align ATR to 4h timeframe (using previous day's ATR for breakout calculation)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Get previous day's close for breakout levels
    prev_close_1d = np.roll(close_1d, 1)
    prev_close_1d[0] = close_1d[0]  # First value
    prev_close_aligned = align_htf_to_ltf(prices, df_1d, prev_close_1d)
    
    # Calculate breakout levels: previous close ± 0.5 * ATR
    upper_break = prev_close_aligned + 0.5 * atr_14_aligned
    lower_break = prev_close_aligned - 0.5 * atr_14_aligned
    
    # Get 4h EMA50 for trend filter
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation (20-period MA on 4h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA50 (50), volume MA (20), and valid ATR
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50[i]) or 
            np.isnan(upper_break[i]) or 
            np.isnan(lower_break[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter
        uptrend = close[i] > ema_50[i]
        downtrend = close[i] < ema_50[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: uptrend + price breaks above upper level + volume
            if uptrend and close[i] > upper_break[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: downtrend + price breaks below lower level + volume
            elif downtrend and close[i] < lower_break[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend breaks or price re-enters below EMA50
            if not uptrend or close[i] < ema_50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend breaks or price re-enters above EMA50
            if not downtrend or close[i] > ema_50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals