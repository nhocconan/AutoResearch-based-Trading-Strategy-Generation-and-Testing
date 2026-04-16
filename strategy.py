#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 12h volume filter and chop regime
# Long when price breaks above Camarilla R1 + 12h volume > 1.5x 20-period avg + chop < 61.8 (trending regime)
# Short when price breaks below Camarilla S1 + 12h volume > 1.5x 20-period avg + chop < 61.8
# Uses discrete position sizing (0.25) to control drawdown and minimize fee drag.
# Camarilla levels provide intraday support/resistance that work in ranging and trending markets.
# Volume filter targets ~30-50 trades/year on 4h timeframe to avoid overtrading.
# Chop regime filter (chop < 61.8) ensures we only trade in trending markets, avoiding whipsaws in ranges.

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
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # === 12h Indicator: Volume SMA ===
    vol_12h = df_12h['volume'].values
    vol_sma_20_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_sma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_sma_20_12h)
    
    # === 4h Camarilla Pivot Levels (based on previous day) ===
    # Typical Price = (high + low + close) / 3
    typical_price = (high + low + close) / 3.0
    # Daily typical price (using 4h data, we approximate daily by looking back 6 periods)
    tp_series = pd.Series(typical_price)
    # For 4h timeframe, daily lookback is 6 periods (6 * 4h = 24h)
    daily_tp = tp_series.rolling(window=6, min_periods=6).mean().shift(6)  # previous day's typical price
    daily_high = pd.Series(high).rolling(window=6, min_periods=6).max().shift(6)
    daily_low = pd.Series(low).rolling(window=6, min_periods=6).min().shift(6)
    daily_close = pd.Series(close).rolling(window=6, min_periods=6).mean().shift(6)
    
    # Camarilla levels
    camarilla_r1 = daily_close + (daily_high - daily_low) * 1.1 / 12
    camarilla_s1 = daily_close - (daily_high - daily_low) * 1.1 / 12
    
    # === 4h Choppiness Index (CHOP) ===
    # True Range
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    # ATR(14)
    atr_14 = tr.rolling(window=14, min_periods=14).mean()
    # Sum of TR over 14 periods
    sum_tr_14 = tr.rolling(window=14, min_periods=14).sum()
    # Choppiness Index: CHOP = 100 * log10(sum_tr_14 / (atr_14 * 14)) / log10(14)
    chop = 100 * np.log10(sum_tr_14 / (atr_14 * 14)) / np.log10(14)
    chop_values = chop.values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(20, 14, 6) + 5  # volume(20), chop(14), camarilla(6) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r1[i]) or np.isnan(camarilla_s1[i]) or
            np.isnan(vol_sma_20_12h_aligned[i]) or np.isnan(chop_values[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current 12h volume > 1.5x 20-period 12h volume SMA
        vol_confirm = vol_12h[i // 48] > (vol_sma_20_12h_aligned[i] * 1.5) if i // 48 < len(vol_12h) else False
        
        # Chop regime filter: chop < 61.8 (trending market)
        chop_filter = chop_values[i] < 61.8
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Camarilla R1
        # 2. Volume confirmation
        # 3. Chop regime filter (trending market)
        if (close[i] > camarilla_r1[i]) and vol_confirm and chop_filter:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Camarilla S1
        # 2. Volume confirmation
        # 3. Chop regime filter (trending market)
        elif (close[i] < camarilla_s1[i]) and vol_confirm and chop_filter:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_Camarilla_R1S1_12hVol_Chop_Filter_v1"
timeframe = "4h"
leverage = 1.0