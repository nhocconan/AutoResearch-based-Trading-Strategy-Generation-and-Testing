#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h HMA21 trend filter and volume confirmation
# Long when price breaks above 4h Donchian upper band + 12h close > 12h HMA21 (uptrend) + volume > 2.0x 20-period avg
# Short when price breaks below 4h Donchian lower band + 12h close < 12h HMA21 (downtrend) + volume > 2.0x 20-period avg
# Uses discrete position sizing (0.25) for balance between return and drawdown control.
# 12h HMA21 provides strong trend filter reducing whipsaws in both bull and bear markets.
# Higher volume threshold (2.0x) ensures only significant breakouts are traded, targeting ~20-30 trades/year.

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
    
    # Get 12h HTF data once before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 55:
        return np.zeros(n)
    
    # === 12h Indicator: HMA21 (trend filter) ===
    def calculate_hma(series, period):
        """Calculate Hull Moving Average"""
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        wma_half = pd.Series(series).ewm(span=half_period, adjust=False).mean()
        wma_full = pd.Series(series).ewm(span=period, adjust=False).mean()
        raw_hma = 2 * wma_half - wma_full
        hma = pd.Series(raw_hma).ewm(span=sqrt_period, adjust=False).mean()
        return hma.values
    
    close_12h = df_12h['close'].values
    hma21_12h = calculate_hma(close_12h, 21)
    hma21_12h_aligned = align_htf_to_ltf(prices, df_12h, hma21_12h)
    
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
    warmup = max(period, 20) + 21  # Donchian(20) + volume(20) + HMA21(12h)
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2.0x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 2.0)
        
        # Skip if any required data is NaN
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or
            np.isnan(hma21_12h_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Get aligned 12h close for trend filter
        close_12h_aligned = align_htf_to_ltf(prices, df_12h, close_12h)[i]
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above upper band (close > upper_band)
        # 2. 12h close > 12h HMA21 (uptrend)
        # 3. Volume confirmation
        if (close[i] > upper_band[i]) and \
           (close_12h_aligned > hma21_12h_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below lower band (close < lower_band)
        # 2. 12h close < 12h HMA21 (downtrend)
        # 3. Volume confirmation
        elif (close[i] < lower_band[i]) and \
             (close_12h_aligned < hma21_12h_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_Donchian20_12hHMA21_Volume_Filter_v1"
timeframe = "4h"
leverage = 1.0