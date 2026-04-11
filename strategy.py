#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4d_adaptive_kama_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return signals
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Calculate 4h KAMA for trend filter
    close_4h = df_4h['close'].values
    change_4h = np.abs(np.diff(close_4h, prepend=close_4h[0]))
    direction_4h = np.abs(np.diff(close_4h, k=10, prepend=close_4h[:10]))
    er_4h = np.where(change_4h != 0, direction_4h / change_4h, 0)
    sc_4h = (er_4h * (0.6667 - 0.0645) + 0.0645) ** 2
    kama_4h = np.zeros_like(close_4h)
    kama_4h[0] = close_4h[0]
    for i in range(1, len(close_4h)):
        kama_4h[i] = kama_4h[i-1] + sc_4h[i] * (close_4h[i] - kama_4h[i-1])
    kama_4h = np.roll(kama_4h, 1)  # shift for completed bar
    kama_4h[0] = np.nan
    
    # Calculate 1d volatility regime (ATR-based)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr_1d = np.maximum(high_1d[1:] - low_1d[1:], 
                       np.maximum(np.abs(high_1d[1:] - close_1d[:-1]),
                                  np.abs(low_1d[1:] - close_1d[:-1])))
    tr_1d = np.concatenate([[np.nan], tr_1d])
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_ma_1d = pd.Series(atr_1d).rolling(window=50, min_periods=50).mean().values
    vol_regime = atr_1d / atr_ma_1d  # >1 = high vol, <1 = low vol
    vol_regime = np.roll(vol_regime, 1)  # shift for completed bar
    vol_regime[0] = np.nan
    
    # Calculate 1h KAMA for entry signal
    change_1h = np.abs(np.diff(close, prepend=close[0]))
    direction_1h = np.abs(np.diff(close, k=10, prepend=close[:10]))
    er_1h = np.where(change_1h != 0, direction_1h / change_1h, 0)
    sc_1h = (er_1h * (0.6667 - 0.0645) + 0.0645) ** 2
    kama_1h = np.zeros_like(close)
    kama_1h[0] = close[0]
    for i in range(1, len(close)):
        kama_1h[i] = kama_1h[i-1] + sc_1h[i] * (close[i] - kama_1h[i-1])
    
    # Align 4h indicators to 1h timeframe
    kama_4h_aligned = align_htf_to_ltf(prices, df_4h, kama_4h)
    vol_regime_aligned = align_htf_to_ltf(prices, df_1d, vol_regime)
    
    # Volume filter: volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(kama_4h_aligned[i]) or np.isnan(vol_regime_aligned[i]) or
            np.isnan(kama_1h[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Session filter
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        kama_4h_val = kama_4h_aligned[i]
        vol_reg = vol_regime_aligned[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.3 * vol_ma
        
        # Trend filter: price vs 4h KAMA
        trend_up = price_close > kama_4h_val
        trend_down = price_close < kama_4h_val
        
        # Volatility regime filter: only trade in low volatility (mean reversion favorable)
        low_volatility = vol_reg < 1.2
        
        # Mean reversion entry: price deviates from 1h KAMA
        kama_dev = (price_close - kama_1h[i]) / kama_1h[i]
        long_signal = volume_confirmed and trend_down and low_volatility and (kama_dev < -0.008)
        short_signal = volume_confirmed and trend_up and low_volatility and (kama_dev > 0.008)
        
        # Exit when price returns to KAMA or volatility increases
        exit_long = position == 1 and (kama_dev >= -0.002 or vol_reg > 1.5)
        exit_short = position == -1 and (kama_dev <= 0.002 or vol_reg > 1.5)
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Adaptive KAMA-based mean reversion on 1h with 4h trend filter and 1d volatility regime.
# Uses Kaufman's Adaptive Moving Average (KAMA) which adapts to market noise - 
# faster in trending markets, slower in ranging markets. 
# 4h KAMA determines trend direction (only trade mean reversion against higher timeframe trend).
# 1d ATR volatility regime filter: only trade in low volatility environments where mean reversion works.
# 1h KAMA deviation triggers entries when price extends too far from adaptive mean.
# Volume confirmation ensures institutional participation.
# Session filter (08-20 UTC) reduces noise during low liquidity periods.
# Position size fixed at 0.20 to manage risk and minimize churn.
# Target: 60-150 total trades over 4 years (15-37/year) to avoid fee drag on 1h timeframe.