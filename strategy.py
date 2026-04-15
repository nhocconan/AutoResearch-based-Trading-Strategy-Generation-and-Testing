#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Donchian breakout with 4h EMA21 trend filter and volume spike
# Long when price breaks above 4h Donchian upper (20) + 4h EMA21 uptrend + volume > 2.0x 20-period avg
# Short when price breaks below 4h Donchian lower (20) + 4h EMA21 downtrend + volume > 2.0x 20-period avg
# Uses 4h for signal direction (structure/trend) and 1h only for entry timing precision
# Session filter (08-20 UTC) to avoid low-liquidity hours
# Discrete position sizing 0.20 to control drawdown and minimize fee churn
# Target: 60-150 total trades over 4 years (~15-37/year) to avoid fee drag

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
    
    # Get 4h HTF data once before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # === 4h Indicators: Donchian(20) and EMA21 ===
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Donchian channels (20-period) on 4h
    donchian_high_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # EMA21 on 4h close
    ema_21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align 4h indicators to 1h timeframe (wait for completed 4h bar)
    donchian_high_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_4h)
    donchian_low_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_4h)
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # Volume SMA for confirmation (20-period on 1h)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(20, 21) + 5  # Donchian(20) + EMA21 + volume(20) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_4h_aligned[i]) or np.isnan(donchian_low_4h_aligned[i]) or
            np.isnan(ema_21_4h_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2.0x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 2.0)
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 4h Donchian upper (close > upper)
        # 2. 4h EMA21 uptrend (close > EMA21)
        # 3. Volume confirmation
        if (close[i] > donchian_high_4h_aligned[i]) and \
           (close[i] > ema_21_4h_aligned[i]) and vol_confirm:
            signals[i] = 0.20
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 4h Donchian lower (close < lower)
        # 2. 4h EMA21 downtrend (close < EMA21)
        # 3. Volume confirmation
        elif (close[i] < donchian_low_4h_aligned[i]) and \
             (close[i] < ema_21_4h_aligned[i]) and vol_confirm:
            signals[i] = -0.20
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1h_Donchian20_4hEMA21_Volume_Filter_v1"
timeframe = "1h"
leverage = 1.0