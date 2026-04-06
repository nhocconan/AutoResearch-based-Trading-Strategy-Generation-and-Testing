#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Triple EMA crossover with 12-hour volume confirmation and 1-day ATR volatility filter.
# Uses three EMAs (8, 21, 55) for trend detection - avoids whipsaws by requiring alignment.
# Volume filter ensures institutional participation on breakouts.
# ATR filter avoids low-volatility periods where false breakouts occur.
# Designed for 4h timeframe targeting 75-200 trades over 4 years with controlled frequency.

name = "4h_tripleema12h_vol1d_atr_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12-hour EMA(8), EMA(21), EMA(55) for trend alignment
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMAs on 12h data
    ema8_12h = pd.Series(close_12h).ewm(span=8, adjust=False).mean().values
    ema21_12h = pd.Series(close_12h).ewm(span=21, adjust=False).mean().values
    ema55_12h = pd.Series(close_12h).ewm(span=55, adjust=False).mean().values
    
    # Align to 4h timeframe
    ema8_12h_aligned = align_htf_to_ltf(prices, df_12h, ema8_12h)
    ema21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema21_12h)
    ema55_12h_aligned = align_htf_to_ltf(prices, df_12h, ema55_12h)
    
    # 1-day volume average for confirmation
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # 1-day ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First period
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period (max of EMA55 needs 55, ATR needs 14, vol needs 20)
    start = max(55, 14, 20)
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(ema8_12h_aligned[i]) or np.isnan(ema21_12h_aligned[i]) or 
            np.isnan(ema55_12h_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or
            np.isnan(atr_1d_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.5x daily average
        volume_filter = volume[i] > vol_ma_1d_aligned[i] * 1.5
        
        # Volatility filter: ATR > 50-period average to avoid low-vol whipsaws
        if i >= 50:
            atr_ma = np.nanmean(atr_1d_aligned[i-50:i])
            vol_filter = atr_1d_aligned[i] > atr_ma * 0.5
        else:
            vol_filter = True  # Not enough data for MA, allow trade
        
        # EMA alignment conditions
        ema_bullish = ema8_12h_aligned[i] > ema21_12h_aligned[i] > ema55_12h_aligned[i]
        ema_bearish = ema8_12h_aligned[i] < ema21_12h_aligned[i] < ema55_12h_aligned[i]
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: EMA bearish alignment or stoploss
            if (ema_bearish or 
                close[i] < entry_price - 2.0 * atr_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: EMA bullish alignment or stoploss
            if (ema_bullish or 
                close[i] > entry_price + 2.0 * atr_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume and volatility confirmation
            if volume_filter and vol_filter:
                # Long: bullish EMA alignment
                if ema_bullish:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: bearish EMA alignment
                elif ema_bearish:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals