#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d ATR volatility filter and volume confirmation
# Long when price breaks above 12h Donchian upper (20-period) + 1d ATR(14) < 0.03*close + volume > 1.5x 20-period avg
# Short when price breaks below 12h Donchian lower (20-period) + 1d ATR(14) < 0.03*close + volume > 1.5x 20-period avg
# Uses discrete position sizing (0.25) to minimize fee drag and control drawdown.
# 1d ATR filter avoids high volatility periods where breakouts fail, improving win rate in both bull and bear markets.
# Volume threshold (1.5x) targets ~20-40 trades/year to minimize fee drag on 12h timeframe.
# Donchian channels provide clear structure with proven edge on SOLUSDT and adaptability to BTC/ETH.

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
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # === 1d Indicator: ATR(14) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]  # first bar has no previous close
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR as EMA of TR (Wilder's smoothing)
    atr_14_1d = np.full_like(tr, np.nan, dtype=np.float64)
    atr_14_1d[13] = np.mean(tr[:14])  # seed with SMA
    for i in range(14, len(tr)):
        atr_14_1d[i] = (atr_14_1d[i-1] * 13 + tr[i]) / 14
    
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # === 12h Donchian Channels (20-period) ===
    # Upper = max(high, 20), Lower = min(low, 20)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(20, 20) + 5  # Donchian(20) + volume(20) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(atr_14_1d_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Volatility filter: 1d ATR(14) < 3% of price (low volatility regime)
        vol_filter = atr_14_1d_aligned[i] < (0.03 * close[i])
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Donchian upper (close > upper)
        # 2. Low volatility regime (ATR filter)
        # 3. Volume confirmation
        if (close[i] > donchian_upper[i]) and vol_filter and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Donchian lower (close < lower)
        # 2. Low volatility regime (ATR filter)
        # 3. Volume confirmation
        elif (close[i] < donchian_lower[i]) and vol_filter and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Donchian20_1dATR14_Volume_Filter_v1"
timeframe = "12h"
leverage = 1.0