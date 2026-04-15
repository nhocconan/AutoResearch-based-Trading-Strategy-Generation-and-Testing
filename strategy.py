#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation
# Long when price breaks above 4h Donchian upper (20-period) + 1d EMA34 > 1d EMA89 (uptrend) + volume > 1.5x 20-period avg
# Short when price breaks below 4h Donchian lower (20-period) + 1d EMA34 < 1d EMA89 (downtrend) + volume > 1.5x 20-period avg
# Uses discrete position sizing (0.25) to minimize fee churn. Designed for low trade frequency (20-40/year).
# EMA trend filter (34/89) adapts to bull/bear markets by requiring alignment with longer-term EMA.
# Works in bull markets (price > EMA34 > EMA89) and bear markets (price < EMA34 < EMA89) by requiring EMA alignment.

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
    if len(df_1d) < 90:
        return np.zeros(n)
    
    # === 1d Indicator: EMA34 and EMA89 (trend filter) ===
    close_1d = df_1d['close'].values
    
    # EMA34
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    # EMA89
    ema89_1d = pd.Series(close_1d).ewm(span=89, adjust=False, min_periods=89).mean().values
    
    # EMA trend: 1 if EMA34 > EMA89 (uptrend), -1 if EMA34 < EMA89 (downtrend), 0 otherwise
    ema_trend = np.where(ema34_1d > ema89_1d, 1, np.where(ema34_1d < ema89_1d, -1, 0))
    ema_trend_aligned = align_htf_to_ltf(prices, df_1d, ema_trend)
    
    # === 4h Indicator: Donchian Channel (20-period) ===
    donchian_window = 20
    donchian_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(donchian_window, 89) + 20  # Donchian(20) + EMA89 + volume(20)
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema_trend_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 4h Donchian upper (20-period)
        # 2. Uptrend (1d EMA34 > EMA89)
        # 3. Volume confirmation
        if (close[i] > donchian_high[i]) and \
           (ema_trend_aligned[i] == 1) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 4h Donchian lower (20-period)
        # 2. Downtrend (1d EMA34 < EMA89)
        # 3. Volume confirmation
        elif (close[i] < donchian_low[i]) and \
             (ema_trend_aligned[i] == -1) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_Donchian20_1dEMA34_89_Volume_Filter_v1"
timeframe = "4h"
leverage = 1.0