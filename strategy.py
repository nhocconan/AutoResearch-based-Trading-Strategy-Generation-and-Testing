#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d ATR(14) volatility filter and volume confirmation
# Long when price breaks above 12h Donchian upper + 12h close > 12h open (bullish candle) + volume > 1.5x 20-period avg + ATR(14) > 0.5 * ATR(50)
# Short when price breaks below 12h Donchian lower + 12h close < 12h open (bearish candle) + volume > 1.5x 20-period avg + ATR(14) > 0.5 * ATR(50)
# Uses discrete position sizing (0.25) to minimize fee drag and control drawdown.
# ATR volatility filter ensures trades occur during sufficient volatility regimes, reducing whipsaws in low-volatility periods.
# Volume threshold (1.5x) targets ~20-40 trades/year to minimize fee drag on 12h timeframe.
# Donchian channels calculated from 12h high/low over 20 periods.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d HTF data once before loop for ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d Indicators: ATR(14) and ATR(50) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First period TR
    tr2[0] = np.abs(high_1d[0] - close_1d[0])  # First period TR (no previous close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Align ATR arrays to 12h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_50_aligned = align_htf_to_ltf(prices, df_1d, atr_50)
    
    # === 12h Donchian Channels (20-period) ===
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(50, 20) + 5  # ATR(50) + Donchian(20) + volume(20) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(atr_14_aligned[i]) or np.isnan(atr_50_aligned[i]) or
            np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # ATR volatility filter: ATR(14) > 0.5 * ATR(50) ensures sufficient volatility
        vol_filter = atr_14_aligned[i] > (0.5 * atr_50_aligned[i])
        
        # Bullish/bearish candle filter
        bullish_candle = close[i] > open_price[i]
        bearish_candle = close[i] < open_price[i]
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Donchian upper (close > upper)
        # 2. Bullish 12h candle
        # 3. Volume confirmation
        # 4. Sufficient volatility
        if (close[i] > donchian_upper[i]) and \
           bullish_candle and vol_confirm and vol_filter:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Donchian lower (close < lower)
        # 2. Bearish 12h candle
        # 3. Volume confirmation
        # 4. Sufficient volatility
        elif (close[i] < donchian_lower[i]) and \
             bearish_candle and vol_confirm and vol_filter:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Donchian20_ATR_Volume_Filter_v1"
timeframe = "12h"
leverage = 1.0