#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 12h ADX regime filter
# Long when Bull Power > 0 (close > EMA13) AND Bear Power < 0 (low < EMA13) AND 12h ADX > 25 (strong trend)
# Short when Bull Power < 0 AND Bear Power > 0 AND 12h ADX > 25
# Uses discrete sizing (0.25) for low turnover. Works in trending markets (bull/bear) by filtering with 12h ADX.
# Avoids ranging markets where Elder Ray whipsaws. Targets 12-30 trades/year.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h HTF data once before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 40:
        return np.zeros(n)
    
    # === 12h Indicator: ADX (14-period) for regime filter ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # Directional Movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smooth(x, period):
        x = np.asarray(x)
        ema = np.full_like(x, np.nan, dtype=float)
        if len(x) >= period:
            # first value: simple average
            ema[period-1] = np.nanmean(x[:period])
            # rest: EMA with alpha=1/period
            alpha = 1.0 / period
            for i in range(period, len(x)):
                if not np.isnan(x[i]) and not np.isnan(ema[i-1]):
                    ema[i] = alpha * x[i] + (1 - alpha) * ema[i-1]
        return ema
    
    atr = wilders_smooth(tr, 14)
    dm_plus_smooth = wilders_smooth(dm_plus, 14)
    dm_minus_smooth = wilders_smooth(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smooth(dx, 14)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # === 6h Indicators: Elder Ray (Bull Power, Bear Power) ===
    # EMA13 of close
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    bull_power = close - ema13
    bear_power = low - ema13
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(13, 14 + 14 + 14)  # EMA13 + ADX smoothing periods
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(adx_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when 12h ADX > 25 (strong trend)
        if adx_12h_aligned[i] <= 25:
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # Bull Power > 0 (close > EMA13) AND Bear Power < 0 (low < EMA13)
        if (bull_power[i] > 0) and (bear_power[i] < 0):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # Bull Power < 0 (close < EMA13) AND Bear Power > 0 (low > EMA13)
        elif (bull_power[i] < 0) and (bear_power[i] > 0):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_ElderRay_12hADX25_Regime_Filter_v1"
timeframe = "6h"
leverage = 1.0