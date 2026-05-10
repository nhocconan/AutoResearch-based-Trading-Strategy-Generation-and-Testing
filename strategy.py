#!/usr/bin/env python3
# 6h_ElderRay_Alligator_Trend_1dRegime
# Hypothesis: Combines Elder Ray (Bull/Bear Power) with Williams Alligator on 6h timeframe.
# Uses 1d timeframe for regime filtering (ADX > 25 for trending, < 20 for ranging).
# In trending markets (ADX > 25): Elder Ray confirms trend strength (Bull Power > 0 for long, Bear Power < 0 for short).
# In ranging markets (ADX < 20): Fades at extremes using Bollinger Bands (20,2) with RSI < 30 for long, > 70 for short.
# Williams Alligator (JAWS=13, TEETH=8, LIPS=5) filters out false signals when lines are intertwined.
# Designed for low trade frequency (~20-40/year) to minimize fee drag and work in both bull/bear markets.
# Target: 15-25 trades/year on 6h timeframe.

name = "6h_ElderRay_Alligator_Trend_1dRegime"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def ema_np(arr, period):
    """Exponential Moving Average using numpy for efficiency."""
    return pd.Series(arr).ewm(span=period, adjust=False, min_periods=period).mean().values

def rsi_np(close, period):
    """Relative Strength Index."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def bollinger_bands(close, period, std_dev):
    """Bollinger Bands."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    return upper, lower

def williams_alligator(high, low, close):
    """Williams Alligator: JAWS (13,8), TEETH (8,5), LIPS (5,3)."""
    median_price = (high + low) / 2
    jaws = ema_np(median_price, 13)  # Blue line
    teeth = ema_np(median_price, 8)   # Red line
    lips = ema_np(median_price, 5)    # Green line
    return jaws, teeth, lips

def elder_ray(high, low, close, ema_period):
    """Elder Ray: Bull Power = High - EMA, Bear Power = Low - EMA."""
    ema = ema_np(close, ema_period)
    bull_power = high - ema
    bear_power = low - ema
    return bull_power, bear_power

def adx(high, low, close, period):
    """Average Directional Index."""
    plus_dm = np.zeros_like(high)
    minus_dm = np.zeros_like(high)
    tr = np.zeros_like(high)
    
    for i in range(1, len(high)):
        plus_dm[i] = max(0, high[i] - high[i-1]) if high[i] - high[i-1] > high[i-1] - low[i-1] else 0
        minus_dm[i] = max(0, low[i-1] - low[i]) if high[i-1] - low[i-1] < high[i] - low[i] else 0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values / (atr + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for regime (ADX)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX for regime filtering
    adx_1d = adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 6h data for signals
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Williams Alligator on 6h
    jaws, teeth, lips = williams_alligator(high, low, close)
    
    # Elder Ray on 6h (EMA=13)
    bull_power, bear_power = elder_ray(high, low, close, 13)
    
    # Bollinger Bands for ranging markets (20,2)
    bb_upper, bb_lower = bollinger_bands(close, 20, 2)
    
    # RSI for ranging markets
    rsi = rsi_np(close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Alligator (13), Elder Ray (13), BB (20), RSI (14), ADX (14)
    start_idx = max(13, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or
            np.isnan(rsi[i]) or np.isnan(adx_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: ADX > 25 = trending, ADX < 20 = ranging
        is_trending = adx_1d_aligned[i] > 25
        is_ranging = adx_1d_aligned[i] < 20
        
        if position == 0:
            # LONG ENTRY
            if is_trending:
                # In trending market: Elder Ray + Alligator alignment (Lips > Teeth > Jaws)
                if bull_power[i] > 0 and lips[i] > teeth[i] and teeth[i] > jaws[i]:
                    signals[i] = 0.25
                    position = 1
            elif is_ranging:
                # In ranging market: Fade at BB lower with RSI < 30
                if close[i] <= bb_lower[i] and rsi[i] < 30:
                    signals[i] = 0.25
                    position = 1
            
            # SHORT ENTRY
            if is_trending:
                # In trending market: Elder Ray + Alligator alignment (Lips < Teeth < Jaws)
                if bear_power[i] < 0 and lips[i] < teeth[i] and teeth[i] < jaws[i]:
                    signals[i] = -0.25
                    position = -1
            elif is_ranging:
                # In ranging market: Fade at BB upper with RSI > 70
                if close[i] >= bb_upper[i] and rsi[i] > 70:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # LONG EXIT
            if is_trending:
                # Exit when trend weakens: Bear Power > 0 or Alligator reverses
                if bear_power[i] > 0 or lips[i] < teeth[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # ranging
                # Exit when price reaches BB middle or RSI > 50
                bb_middle = (bb_upper[i] + bb_lower[i]) / 2
                if close[i] >= bb_middle or rsi[i] > 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        
        elif position == -1:
            # SHORT EXIT
            if is_trending:
                # Exit when trend weakens: Bull Power < 0 or Alligator reverses
                if bull_power[i] < 0 or lips[i] > teeth[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:  # ranging
                # Exit when price reaches BB middle or RSI < 50
                bb_middle = (bb_upper[i] + bb_lower[i]) / 2
                if close[i] <= bb_middle or rsi[i] < 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals