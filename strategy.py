#!/usr/bin/env python3
# 6h_keltner_breakout_1d_atr_v1
# Hypothesis: 6h Keltner Channel breakout with 1d ATR regime filter and volume confirmation.
# Uses 6h timeframe to balance trade frequency and responsiveness. Keltner Channel breakouts capture volatility expansion moves.
# 1d ATR regime filter ensures we only trade when daily volatility is elevated (ATR > 20-period MA), avoiding low-volatility chop.
# Volume confirmation ensures institutional participation. Designed for 12-37 trades/year (50-150 over 4 years).
# Works in bull/bear markets: breakouts work in trending markets, ATR filter avoids false signals in ranging markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_keltner_breakout_1d_atr_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for Keltner Channel calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 30:  # Need enough for EMA(20) and ATR(10)
        return np.zeros(n)
    
    # Calculate 6h EMA(20) for Keltner Channel middle line
    close_6h = df_6h['close'].values
    ema_20 = pd.Series(close_6h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 6h ATR(10) for Keltner Channel width
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    tr1 = high_6h - low_6h
    tr2 = np.abs(high_6h - np.roll(close_6h, 1))
    tr3 = np.abs(low_6h - np.roll(close_6h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR is just high-low
    atr_10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Keltner Channel bands: EMA(20) ± 2 * ATR(10)
    upper_band = ema_20 + 2 * atr_10
    lower_band = ema_20 - 2 * atr_10
    
    # Align 6h Keltner Channel to 6h timeframe (completed 6h candle only)
    ema_20_aligned = align_htf_to_ltf(prices, df_6h, ema_20)
    upper_band_aligned = align_htf_to_ltf(prices, df_6h, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_6h, lower_band)
    
    # Get 1d HTF data ONCE before loop for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 25:  # Need enough for ATR(20) calculation
        return np.zeros(n)
    
    # Calculate 1d ATR(20) for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    tr_1d[0] = tr1_1d[0]  # First period TR is just high-low
    atr_20_1d = pd.Series(tr_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 1d ATR(20) moving average for regime filter
    atr_ma_20_1d = pd.Series(atr_20_1d).rolling(window=20, min_periods=20).mean().values
    
    # ATR regime: trade only when daily ATR > 20-period MA (elevated volatility)
    atr_regime = atr_20_1d > atr_ma_20_1d
    
    # Align 1d ATR regime to 6h timeframe (completed daily candle only)
    atr_regime_aligned = align_htf_to_ltf(prices, df_1d, atr_regime.astype(float))
    
    # Volume spike detection (20-period volume average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_20_aligned[i]) or np.isnan(upper_band_aligned[i]) or 
            np.isnan(lower_band_aligned[i]) or np.isnan(atr_regime_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below 6h EMA(20) (middle line)
            if close[i] < ema_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above 6h EMA(20) (middle line)
            if close[i] > ema_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price closes above 6h Keltner upper band, with ATR regime and volume spike
            if (close[i] > upper_band_aligned[i]) and atr_regime_aligned[i] and vol_spike[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below 6h Keltner lower band, with ATR regime and volume spike
            elif (close[i] < lower_band_aligned[i]) and atr_regime_aligned[i] and vol_spike[i]:
                position = -1
                signals[i] = -0.25
    
    return signals