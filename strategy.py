#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Long when price breaks above 4h Donchian upper band (20-period high) + 1w close > 1w EMA50 (uptrend) + volume > 1.5x 20-period avg
# Short when price breaks below 4h Donchian lower band (20-period low) + 1w close < 1w EMA50 (downtrend) + volume > 1.5x 20-period avg
# Uses discrete position sizing (0.25) to reduce fee churn and improve drawdown resistance.
# Weekly EMA50 provides a stronger, more stable trend filter than daily, reducing whipsaws in ranging markets.
# Volume confirmation avoids false breakouts. Designed for low trade frequency (15-30/year) to minimize fee drag.

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
    
    # Get 1w HTF data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # === 1w Indicator: EMA50 (trend filter) ===
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # === 4h Donchian Channel (20-period) ===
    period = 20
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    upper_band = highest_high
    lower_band = lowest_low
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(period, 20) + 50  # Donchian(20) + volume(20) + EMA50(1w)
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Get aligned 1w close for trend filter
        close_1w_aligned = align_htf_to_ltf(prices, df_1w, close_1w)[i]
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above upper band (close > upper_band)
        # 2. 1w close > 1w EMA50 (uptrend)
        # 3. Volume confirmation
        if (close[i] > upper_band[i]) and \
           (close_1w_aligned > ema50_1w_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below lower band (close < lower_band)
        # 2. 1w close < 1w EMA50 (downtrend)
        # 3. Volume confirmation
        elif (close[i] < lower_band[i]) and \
             (close_1w_aligned < ema50_1w_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_Donchian20_1wEMA50_Volume_Filter_v1"
timeframe = "4h"
leverage = 1.0