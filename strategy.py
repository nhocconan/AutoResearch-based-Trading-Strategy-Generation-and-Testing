#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Elder Ray Index with 1d ADX regime filter and volume confirmation
# - Elder Ray measures bull/bear power via EMA(13) for trend-following entries
# - Uses 1d ADX(14) > 25 to filter for trending regimes only (avoids chop)
# - Requires volume > 1.5 * 20-period average for confirmation
# - Long when Bull Power > 0 and Bear Power < 0 with volume confirmation
# - Short when Bear Power < 0 and Bull Power < 0 with volume confirmation
# - ATR(14) stoploss at 2.5 * ATR for risk management
# - Target: 15-30 trades/year on 4h timeframe (60-120 total over 4 years) to avoid fee drag
# - Works in bull markets via Elder Ray buy signals, in bear via sell signals with ADX filter

name = "4h_1d_elder_ray_adx_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d ADX(14) for regime filtering
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    
    # Smoothed TR, DM+, DM-
    tr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Pre-compute 4h EMA(13) for Elder Ray
    close = prices['close'].values
    ema_13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Pre-compute 4h ATR(14) for stoploss
    high = prices['high'].values
    low = prices['low'].values
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute volume confirmation: volume > 1.5 * 20-period average
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_13[i]) or np.isnan(atr[i]) or atr[i] <= 0 or
            np.isnan(adx_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Elder Ray components
        bull_power = high[i] - ema_13[i]
        bear_power = low[i] - ema_13[i]
        
        if position == 1:  # Long position
            # Exit conditions: stoploss or trend reversal
            if close[i] < ema_13[i] - 2.5 * atr[i]:  # ATR stoploss
                position = 0
                signals[i] = 0.0
            elif bull_power <= 0 or bear_power >= 0:  # Trend weakening
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: stoploss or trend reversal
            if close[i] > ema_13[i] + 2.5 * atr[i]:  # ATR stoploss
                position = 0
                signals[i] = 0.0
            elif bull_power >= 0 or bear_power <= 0:  # Trend weakening
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Elder Ray entries with volume confirmation and ADX filter
            if (adx_aligned[i] > 25 and  # Trending regime
                volume_confirm[i] and    # Volume confirmation
                bull_power > 0 and       # Bullish pressure
                bear_power < 0):         # Bearish pressure (confirm trend)
                position = 1
                signals[i] = 0.25
            elif (adx_aligned[i] > 25 and   # Trending regime
                  volume_confirm[i] and     # Volume confirmation
                  bear_power < 0 and        # Bearish pressure
                  bull_power < 0):          # Confirm bearish trend (no bullish pressure)
                position = -1
                signals[i] = -0.25
    
    return signals