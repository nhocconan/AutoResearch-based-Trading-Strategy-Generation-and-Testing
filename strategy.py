#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and ATR-based trend filter
# Long when price breaks above 20-period Donchian high + volume > 1.5x 20-period avg + ATR(14) > ATR(50) (trending volatility)
# Short when price breaks below 20-period Donchian low + volume > 1.5x 20-period avg + ATR(14) > ATR(50)
# Uses discrete position sizing (0.25) to control drawdown and minimize fee drag
# Target: 75-150 total trades over 4 years (19-37/year) to avoid fee drag
# Donchian breakouts work in both bull and bear markets; volatility filter ensures we only trade when trends have strength
# Volume confirmation reduces false breakouts

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
    
    # === Primary Indicators: Donchian Channels (20-period) ===
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # === Volume SMA for confirmation (20-period) ===
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === ATR-based Trend Filter: ATR(14) > ATR(50) indicates increasing volatility/trend strength ===
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # first period has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    volatility_filter = atr_14 > atr_50  # trending when short-term ATR > long-term ATR
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    # Need Donchian(20) + volume(20) + ATR(50) + buffer
    warmup = 60
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_sma_20[i]) or np.isnan(atr_14[i]) or np.isnan(atr_50[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Donchian high (20-period)
        # 2. Volume confirmation
        # 3. Volatility filter (ATR14 > ATR50) indicates trending market
        if (close[i] > donchian_high[i]) and \
           vol_confirm and volatility_filter[i]:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Donchian low (20-period)
        # 2. Volume confirmation
        # 3. Volatility filter (ATR14 > ATR50) indicates trending market
        elif (close[i] < donchian_low[i]) and \
             vol_confirm and volatility_filter[i]:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_Donchian20_Volume_ATRTrend_Filter_v1"
timeframe = "4h"
leverage = 1.0