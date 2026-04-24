#!/usr/bin/env python3
"""
Hypothesis: 1h volume-weighted RSI with 4h EMA trend and 1d ATR regime filter.
- Primary timeframe: 1h for execution, HTF: 4h for EMA trend, 1d for ATR regime.
- Volume-weighted RSI: RSI calculated using volume-weighted average price (VWAP) to filter noise.
- Trend filter: Only trade in direction of 4h EMA20 (long if EMA20 rising, short if falling).
- Regime filter: Only trade when 1d ATR(14) is below its 50-period MA (low volatility regime).
- Entry: Long when VWAP RSI < 30 in uptrend + low vol regime; Short when VWAP RSI > 70 in downtrend + low vol regime.
- Exit: Opposite RSI signal or regime change (high volatility).
- Discrete signal size: 0.20 to limit drawdown and reduce fee churn.
- Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.
- Works in bull via buying oversold in uptrend, in bear via selling overbought in downtrend.
- Low volatility regime avoids whipsaws in choppy markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate VWAP for RSI
    typical_price = (high + low + close) / 3.0
    vwap = np.cumsum(typical_price * volume) / np.cumsum(volume)
    vwap = np.where(np.cumsum(volume) == 0, typical_price, vwap)  # avoid div by zero
    
    # RSI on VWAP
    delta = np.diff(vwap, prepend=vwap[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_ma = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    loss_ma = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = gain_ma / (loss_ma + 1e-10)
    vwap_rsi = 100 - (100 / (1 + rs))
    
    # Get 4h data for EMA20 trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Get 1d data for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # ATR(14) calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first bar
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_ma_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    low_vol_regime = atr_14 < atr_ma_50  # True when ATR below MA (low volatility)
    low_vol_regime_aligned = align_htf_to_ltf(prices, df_1d, low_vol_regime.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 20, 50)  # RSI, EMA20, ATR MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vwap_rsi[i]) or np.isnan(ema_20_4h_aligned[i]) or 
            np.isnan(low_vol_regime_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Only trade in direction of 4h EMA20 trend and low volatility regime
            if i > 0 and not np.isnan(ema_20_4h_aligned[i-1]):
                ema20_slope = ema_20_4h_aligned[i] - ema_20_4h_aligned[i-1]
                in_low_vol = low_vol_regime_aligned[i] > 0.5  # boolean as float
                
                if ema20_slope > 0 and in_low_vol:  # Uptrend + low vol
                    if vwap_rsi[i] < 30:  # Oversold
                        signals[i] = 0.20
                        position = 1
                elif ema20_slope < 0 and in_low_vol:  # Downtrend + low vol
                    if vwap_rsi[i] > 70:  # Overbought
                        signals[i] = -0.20
                        position = -1
        elif position == 1:
            # Long exit: RSI > 50 or regime change to high volatility
            if vwap_rsi[i] > 50 or low_vol_regime_aligned[i] <= 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: RSI < 50 or regime change to high volatility
            if vwap_rsi[i] < 50 or low_vol_regime_aligned[i] <= 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_VWAP_RSI_4hEMA20_1dATR_Regime_v1"
timeframe = "1h"
leverage = 1.0