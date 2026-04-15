#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and ADX(14) trend filter
# Long when price breaks above 4h Donchian upper + volume > 1.5x 20-period avg + ADX > 25
# Short when price breaks below 4h Donchian lower + volume > 1.5x 20-period avg + ADX > 25
# Uses discrete position sizing (0.25) to minimize fee drag and control drawdown.
# ADX filter ensures we only trade in trending markets, reducing whipsaws in ranging periods.
# Volume confirmation (1.5x) targets ~20-40 trades/year to minimize fee drag on 4h timeframe.
# Donchian channels calculated from 4h high/low over 20 periods.

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
    
    # === 4h Indicators ===
    # Donchian Channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume SMA for confirmation (20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ADX (14-period) for trend strength
    # ADX = 100 * smoothed moving average of |+DI - -DI| / (+DI + -DI)
    # Simplified: using typical ATR-based calculation
    plus_dm = high_series.diff()
    minus_dm = low_series.diff().multiply(-1)
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    tr1 = high_series - low_series
    tr2 = abs(high_series - close.shift(1))
    tr3 = abs(low_series - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(20, 14) + 5  # Donchian(20) + ADX(14) + volume(20) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(vol_sma_20[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Trend filter: ADX > 25 indicates strong trend
        trend_filter = adx[i] > 25
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Donchian upper (close > upper)
        # 2. Volume confirmation
        # 3. Strong trend (ADX > 25)
        if (close[i] > donchian_upper[i]) and vol_confirm and trend_filter:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Donchian lower (close < lower)
        # 2. Volume confirmation
        # 3. Strong trend (ADX > 25)
        elif (close[i] < donchian_lower[i]) and vol_confirm and trend_filter:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_Donchian20_Volume_ADX_Filter_v1"
timeframe = "4h"
leverage = 1.0