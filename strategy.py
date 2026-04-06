#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI(14) mean reversion with 4h EMA(50) trend filter and 1d ADX(14) regime filter.
# In bull/bear markets: fade RSI extremes (RSI<30 long, RSI>70 short) only when 4h EMA(50) aligns with trade direction.
# In ranging markets (ADX<25): fade RSI extremes regardless of 4h EMA(50).
# Uses 1d ADX to distinguish trending vs ranging regimes.
# Target: 60-150 total trades over 4 years (15-37/year) with controlled risk.

name = "1h_rsi14_4hema50_1dadx_regime_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    
    # 4h EMA(50) for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d ADX(14) for regime filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = tr1[0]
    
    # Calculate Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    
    # Smooth TR, DM+, DM- with Wilder's smoothing (alpha = 1/14)
    alpha = 1.0 / 14
    atr_1d = np.zeros_like(tr_1d)
    dm_plus_sm = np.zeros_like(dm_plus)
    dm_minus_sm = np.zeros_like(dm_minus)
    
    atr_1d[0] = tr_1d[0]
    dm_plus_sm[0] = dm_plus[0]
    dm_minus_sm[0] = dm_minus[0]
    
    for i in range(1, len(tr_1d)):
        atr_1d[i] = alpha * tr_1d[i] + (1 - alpha) * atr_1d[i-1]
        dm_plus_sm[i] = alpha * dm_plus[i] + (1 - alpha) * dm_plus_sm[i-1]
        dm_minus_sm[i] = alpha * dm_minus[i] + (1 - alpha) * dm_minus_sm[i-1]
    
    # Avoid division by zero
    di_plus = np.where(atr_1d != 0, 100 * dm_plus_sm / atr_1d, 0)
    di_minus = np.where(atr_1d != 0, 100 * dm_minus_sm / atr_1d, 0)
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx_1d = np.zeros_like(dx)
    adx_1d[0] = dx[0]
    for i in range(1, len(dx)):
        adx_1d[i] = alpha * dx[i] + (1 - alpha) * adx_1d[i-1]
    
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 1h RSI(14) for mean reversion signals
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[0] = gain[0]
    avg_loss[0] = loss[0]
    for i in range(1, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):
        # Skip if required data not available
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(adx_1d_aligned[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Regime filter: ADX < 25 = ranging, ADX >= 25 = trending
        ranging_market = adx_1d_aligned[i] < 25
        
        if position == 1:  # long position
            # Exit: RSI > 50 (mean reversion complete) OR RSI > 70 (overbought)
            if rsi[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: RSI < 50 (mean reversion complete) OR RSI < 30 (oversold)
            if rsi[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries: RSI extremes + regime filter
            if rsi[i] < 30:  # Oversold - potential long
                if ranging_market or close[i] > ema_50_4h_aligned[i]:
                    # In ranging market OR uptrend: long
                    signals[i] = 0.20
                    position = 1
            elif rsi[i] > 70:  # Overbought - potential short
                if ranging_market or close[i] < ema_50_4h_aligned[i]:
                    # In ranging market OR downtrend: short
                    signals[i] = -0.20
                    position = -1
    
    return signals