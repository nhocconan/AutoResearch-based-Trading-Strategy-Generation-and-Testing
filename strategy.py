#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d ATR filter and volume confirmation
# Long when price breaks above Donchian(20) high + 1d ATR > 1.5x 20-period ATR mean + volume > 1.5x 20-period vol SMA
# Short when price breaks below Donchian(20) low + 1d ATR > 1.5x 20-period ATR mean + volume > 1.5x 20-period vol SMA
# Uses ATR as volatility filter to avoid low-volume false breakouts
# Discrete position sizing (0.25) to control drawdown and minimize fee drag
# Target: 75-200 total trades over 4 years (19-50/year)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d HTF data once before loop for ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === 1d Indicator: ATR (14-period) for volatility filter ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # first value has no previous close
    tr2[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 20-period ATR mean for volatility regime filter
    atr_ma_20 = pd.Series(atr_14).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d ATR and ATR MA to 4h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_ma_20_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_20)
    
    # === 4h Indicators: Donchian Channel (20-period) ===
    # Donchian high = rolling max of high over 20 periods
    # Donchian low = rolling min of low over 20 periods
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume SMA for confirmation (20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    # Need 1d ATR (14+20 = 34) + 4h Donchian(20) + volume(20) + buffer
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(atr_14_aligned[i]) or np.isnan(atr_ma_20_aligned[i]) or
            np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: current 1d ATR > 1.5x 20-period ATR mean (avoid low volatility chop)
        vol_filter = atr_14_aligned[i] > (atr_ma_20_aligned[i] * 1.5)
        
        # Volume confirmation: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Donchian(20) high
        # 2. High volatility regime (ATR > 1.5x ATR mean)
        # 3. Volume confirmation
        if (close[i] > donchian_high[i]) and \
           vol_filter and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Donchian(20) low
        # 2. High volatility regime (ATR > 1.5x ATR mean)
        # 3. Volume confirmation
        elif (close[i] < donchian_low[i]) and \
             vol_filter and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_Donchian20_1dATR_Volume_Filter_v1"
timeframe = "4h"
leverage = 1.0