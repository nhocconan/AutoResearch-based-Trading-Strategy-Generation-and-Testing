#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1d volume confirmation and ATR filter
# Long when price breaks above Camarilla R4 (1d) + volume > 1.5x 20-period SMA + ATR(14) > 0.5*ATR(50)
# Short when price breaks below Camarilla S4 (1d) + volume > 1.5x 20-period SMA + ATR(14) > 0.5*ATR(50)
# Uses discrete position sizing (0.25) to minimize fee churn. Designed for low trade frequency (12-25/year).
# Camarilla pivots provide mathematically derived support/resistance levels. Volume confirms breakout strength.
# ATR filter ensures sufficient volatility to avoid whipsaws in low-volume periods.
# Works in bull markets (breakout continuation) and bear markets (strong downside breaks) by requiring volume and volatility.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d Indicator: Camarilla Pivot Levels (R4, S4) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point (PP)
    pp = (high_1d + low_1d + close_1d) / 3.0
    # Calculate Camarilla levels
    camarilla_r4 = pp + ((high_1d - low_1d) * 1.1 / 2.0)
    camarilla_s4 = pp - ((high_1d - low_1d) * 1.1 / 2.0)
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # === 6h Indicators: Volume SMA and ATR ===
    # Volume SMA (20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period) and ATR (50-period) for volatility filter
    high_shift = np.roll(high, 1)
    low_shift = np.roll(low, 1)
    close_shift = np.roll(close, 1)
    high_shift[0] = high[0]
    low_shift[0] = low[0]
    close_shift[0] = close[0]
    
    tr1 = high - low
    tr2 = np.abs(high - close_shift)
    tr3 = np.abs(low - close_shift)
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14)
    atr_period = 14
    atr_14 = np.zeros_like(tr)
    atr_14[atr_period-1] = np.mean(tr[:atr_period])
    for i in range(atr_period, len(tr)):
        atr_14[i] = (atr_14[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # ATR(50)
    atr_long_period = 50
    atr_50 = np.zeros_like(tr)
    if len(tr) >= atr_long_period:
        atr_50[atr_long_period-1] = np.mean(tr[:atr_long_period])
        for i in range(atr_long_period, len(tr)):
            atr_50[i] = (atr_50[i-1] * (atr_long_period-1) + tr[i]) / atr_long_period
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(20, atr_long_period) + 5  # volume(20) + ATR(50) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # ATR filter: ATR(14) > 0.5 * ATR(50) (ensures sufficient volatility)
        atr_filter = atr_14[i] > (0.5 * atr_50[i])
        
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(vol_sma_20[i]) or np.isnan(atr_14[i]) or np.isnan(atr_50[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 1d Camarilla R4
        # 2. Volume confirmation
        # 3. Volatility filter
        if (close[i] > camarilla_r4_aligned[i]) and \
           vol_confirm and atr_filter:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 1d Camarilla S4
        # 2. Volume confirmation
        # 3. Volatility filter
        elif (close[i] < camarilla_s4_aligned[i]) and \
             vol_confirm and atr_filter:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_Camarilla_R4S4_1dVolATR_Filter_v1"
timeframe = "6h"
leverage = 1.0