#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour mean reversion with 4-hour trend filter and daily volatility regime
# Long when: price < BB(20,2) lower band, price > 4h EMA(50), daily ATR ratio < 0.8 (low vol)
# Short when: price > BB(20,2) upper band, price < 4h EMA(50), daily ATR ratio < 0.8
# Exit on BB middle cross or trend reversal
# Stoploss at 2 * ATR(14)
# Position size: 0.20
# Uses 4h for trend direction, 1d for volatility regime, 1h for entry timing
# Target: 80-150 trades over 4 years (20-38/year)

name = "1h_bb_4h_ema_1d_vol_regime_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h data for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1d data for volatility regime (ATR ratio)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily ATR(10)
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr2_1d[0] = tr1_1d[0]
    tr3_1d[0] = tr1_1d[0]
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    atr_1d = pd.Series(tr_1d).rolling(window=10, min_periods=10).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate daily SMA(20) of ATR for regime
    atr_ma_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    atr_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_1d)
    
    # ATR ratio: current ATR / 20-day average ATR (< 0.8 = low volatility regime)
    atr_ratio = atr_1d_aligned / atr_ma_1d_aligned
    
    # 1h Bollinger Bands (20, 2)
    bb_length = 20
    bb_mult = 2.0
    basis = pd.Series(close).rolling(window=bb_length, min_periods=bb_length).mean().values
    dev = bb_mult * pd.Series(close).rolling(window=bb_length, min_periods=bb_length).std().values
    upper_band = basis + dev
    lower_band = basis - dev
    
    # 1h ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):  # Start after warmup
        # Skip if required data not available
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(basis[i]) or np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i]) or np.isnan(atr_ratio[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Low volatility regime filter
        low_vol = atr_ratio[i] < 0.8
        
        if position == 1:  # long position
            # Stoploss: 2 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses above BB middle or trend turns bearish
            elif close[i] > basis[i] or close[i] < ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Stoploss: 2 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses below BB middle or trend turns bullish
            elif close[i] < basis[i] or close[i] > ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.20
        else:
            # Look for mean reversion entries in low volatility regime
            # Long: price < lower BB, price > 4h EMA (bullish trend), low volatility
            if (close[i] < lower_band[i] and
                close[i] > ema_4h_aligned[i] and
                low_vol):
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            # Short: price > upper BB, price < 4h EMA (bearish trend), low volatility
            elif (close[i] > upper_band[i] and
                  close[i] < ema_4h_aligned[i] and
                  low_vol):
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
    
    return signals