#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h ADX(14) trend strength + 12h Williams %R(14) mean reversion + volume confirmation
# Long when: ADX > 25 (trending) AND Williams %R < -80 (oversold) AND volume > 1.5x 20-period average
# Short when: ADX > 25 (trending) AND Williams %R > -20 (overbought) AND volume > 1.5x 20-period average
# Uses discrete position sizing (0.25) to minimize fee drag and control drawdown.
# ADX filters for trending markets only, avoiding whipsaws in ranging conditions.
# Williams %R identifies overextended moves within the trend for mean-reversion entries.
# Volume confirmation ensures participation, targeting ~20-40 trades/year on 6h timeframe.

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
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # === 12h HTF: Williams %R(14) ===
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_12h = pd.Series(df_12h['high']).rolling(window=14, min_periods=14).max().values
    lowest_low_12h = pd.Series(df_12h['low']).rolling(window=14, min_periods=14).min().values
    close_12h = df_12h['close'].values
    
    williams_r_12h = np.full_like(close_12h, np.nan)
    for i in range(len(close_12h)):
        if not (np.isnan(highest_high_12h[i]) or np.isnan(lowest_low_12h[i]) or highest_high_12h[i] == lowest_low_12h[i]):
            williams_r_12h[i] = ((highest_high_12h[i] - close_12h[i]) / (highest_high_12h[i] - lowest_low_12h[i])) * -100
    
    williams_r_12h_aligned = align_htf_to_ltf(prices, df_12h, williams_r_12h)
    
    # === Primary TF: ADX(14) ===
    # ADX calculation: +DI, -DI, DX, then ADX smoothed
    # +DM = max(high - prev_high, 0) if high - prev_high > prev_low - low else 0
    # -DM = max(prev_low - low, 0) if prev_low - low > high - prev_high else 0
    # TR = max(high - low, abs(high - prev_close), abs(low - prev_close))
    # +DI = 100 * smoothed(+DM) / smoothed(TR)
    # -DI = 100 * smoothed(-DM) / smoothed(TR)
    # DX = 100 * abs(+DI - -DI) / (+DI + -DI)
    # ADX = smoothed(DX)
    
    prev_close = np.roll(close, 1)
    prev_close[0] = np.nan
    prev_high = np.roll(high, 1)
    prev_high[0] = np.nan
    prev_low = np.roll(low, 1)
    prev_low[0] = np.nan
    
    plus_dm = np.where((high - prev_high) > (prev_low - low), np.maximum(high - prev_high, 0), 0)
    minus_dm = np.where((prev_low - low) > (high - prev_high), np.maximum(prev_low - low, 0), 0)
    
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smoothing with Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.nansum(data[:period]) / period
            # Subsequent values: smoothed = prev_smoothed * (1 - 1/period) + current * (1/period)
            for i in range(period, len(data)):
                if not np.isnan(data[i]):
                    result[i] = result[i-1] * (1 - 1/period) + data[i] * (1/period)
        return result
    
    period = 14
    plus_di_smoothed = wilders_smoothing(plus_dm, period)
    minus_di_smoothed = wilders_smoothing(minus_dm, period)
    tr_smoothed = wilders_smoothing(tr, period)
    
    plus_di = np.where(tr_smoothed != 0, (plus_di_smoothed / tr_smoothed) * 100, 0)
    minus_di = np.where(tr_smoothed != 0, (minus_di_smoothed / tr_smoothed) * 100, 0)
    
    dx = np.where((plus_di + minus_di) != 0, (np.abs(plus_di - minus_di) / (plus_di + minus_di)) * 100, 0)
    adx = wilders_smoothing(dx, period)
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(period*2, 20) + 5  # ADX needs 2*period for smoothing + volume + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(adx[i]) or np.isnan(williams_r_12h_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # === LONG CONDITIONS ===
        # 1. ADX > 25 (strong trend)
        # 2. Williams %R < -80 (oversold)
        # 3. Volume confirmation
        if (adx[i] > 25) and \
           (williams_r_12h_aligned[i] < -80) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. ADX > 25 (strong trend)
        # 2. Williams %R > -20 (overbought)
        # 3. Volume confirmation
        elif (adx[i] > 25) and \
             (williams_r_12h_aligned[i] > -20) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_ADX14_WilliamsR14_12h_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0