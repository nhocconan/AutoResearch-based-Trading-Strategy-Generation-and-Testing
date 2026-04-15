#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1w trend filter and volume confirmation
# Long when price breaks above 6h Camarilla R4 + 1w close > 1w open (bullish weekly candle) + volume > 1.5x 20-period avg
# Short when price breaks below 6h Camarilla S4 + 1w close < 1w open (bearish weekly candle) + volume > 1.5x 20-period avg
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 50-150 total trades over 4 years.
# Camarilla pivots provide mathematical support/resistance levels. Weekly trend filter ensures we trade with the higher timeframe momentum.
# Volume confirmation adds conviction to breakouts. Works in bull markets (buy strength) and bear markets (sell weakness).

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
    
    # Get 1w HTF data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # === 1w Indicator: Weekly bullish/bearish candle ===
    close_1w = df_1w['close'].values
    open_1w = df_1w['open'].values
    weekly_bullish = close_1w > open_1w  # True for bullish weekly candle
    weekly_bearish = close_1w < open_1w  # True for bearish weekly candle
    
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish.astype(float))
    
    # === 6h Indicator: Camarilla Pivot Levels (based on prior 6h bar) ===
    # Camarilla levels calculated from previous bar's high, low, close
    # R4 = close + ((high - low) * 1.1/2)
    # S4 = close - ((high - low) * 1.1/2)
    # We use the prior completed 6h bar to avoid look-ahead
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    # First bar: use current values (will be filtered out by warmup anyway)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    camarilla_r4 = prev_close + ((prev_high - prev_low) * 1.1 / 2)
    camarilla_s4 = prev_close - ((prev_high - prev_low) * 1.1 / 2)
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 20  # volume SMA(20) + 1 for prior bar
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r4[i]) or np.isnan(camarilla_s4[i]) or
            np.isnan(weekly_bullish_aligned[i]) or np.isnan(weekly_bearish_aligned[i]) or
            np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 6h Camarilla R4
        # 2. Weekly trend bullish (1w close > 1w open)
        # 3. Volume confirmation
        if (close[i] > camarilla_r4[i]) and \
           (weekly_bullish_aligned[i] > 0.5) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 6h Camarilla S4
        # 2. Weekly trend bearish (1w close < 1w open)
        # 3. Volume confirmation
        elif (close[i] < camarilla_s4[i]) and \
             (weekly_bearish_aligned[i] > 0.5) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_CamarillaR4S4_1wTrend_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0