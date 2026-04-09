#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze + 1d ADX trend filter + volume breakout
# - Primary signal: Bollinger Band width at 6h < 20th percentile (squeeze) + price breaks above/below BB(20,2)
# - Trend filter: 1d ADX > 25 ensures we only trade in trending markets (avoid whipsaws in ranges)
# - Volume confirmation: breakout candle volume > 1.5x 20-period average volume
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) per 6h strategy guidelines
# - Works in bull/bear: BB squeeze captures low volatility pre-breakout, ADX filter ensures
#   we only trade when trend is strong, volume confirms institutional participation

name = "6h_1d_bb_squeeze_adx_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d ADX(14) for trend strength
    # +DM = max(high - prev_high, 0) if > max(prev_low - low, 0) else 0
    # -DM = max(prev_low - low, 0) if > max(high - prev_high, 0) else 0
    # TR = max(high - low, abs(high - prev_close), abs(low - prev_close))
    # +DI = 100 * smoothed(+DM) / ATR
    # -DI = 100 * smoothed(-DM) / ATR
    # ADX = 100 * smoothed(|+DI - -DI| / (+DI + -DI))
    
    # Calculate +DM, -DM, TR
    high_diff = np.diff(high_1d, prepend=high_1d[0])
    low_diff = np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((high_diff > -low_diff) & (high_diff > 0), high_diff, 0.0)
    minus_dm = np.where((-low_diff > high_diff) & (-low_diff > 0), -low_diff, 0.0)
    tr1 = high_1d - low_1d
    tr2 = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    tr3 = np.abs(np.diff(close_1d, prepend=close_1d[0])[1:])
    tr3 = np.append(tr3, 0.0)  # align length
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr = wilders_smoothing(tr, 14)
    plus_di = 100 * wilders_smoothing(plus_dm, 14) / np.where(atr == 0, 1, atr)
    minus_di = 100 * wilders_smoothing(minus_dm, 14) / np.where(atr == 0, 1, atr)
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) == 0, 1, (plus_di + minus_di))
    adx = wilders_smoothing(dx, 14)
    
    # Align 1d ADX to 6h timeframe (completed 1d bar only)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 6h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h Bollinger Bands (20, 2)
    basis = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    dev = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_band = basis + 2.0 * dev
    lower_band = basis - 2.0 * dev
    bb_width = (upper_band - lower_band) / np.where(basis == 0, 1, basis) * 100  # as percentage
    
    # 6h BB width percentile (20-period lookback)
    bb_width_percentile = pd.Series(bb_width).rolling(window=100, min_periods=20).rank(pct=True).values
    bb_squeeze = bb_width_percentile < 0.20  # bottom 20% = squeeze
    
    # 6h volume confirmation: volume > 1.5x 20-period average
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * avg_volume_20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_aligned[i]) or
            np.isnan(bb_squeeze[i]) or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below BB middle OR BB squeeze ends (volatility expansion)
            if close[i] < basis[i] or not bb_squeeze[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above BB middle OR BB squeeze ends
            if close[i] > basis[i] or not bb_squeeze[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for BB squeeze breakout with volume and ADX filter
            # Long: price breaks above upper BB AND BB squeeze AND volume confirm AND ADX > 25
            if close[i] > upper_band[i] and bb_squeeze[i] and volume_confirm[i] and adx_aligned[i] > 25:
                position = 1
                signals[i] = 0.25
            # Short: price breaks below lower BB AND BB squeeze AND volume confirm AND ADX > 25
            elif close[i] < lower_band[i] and bb_squeeze[i] and volume_confirm[i] and adx_aligned[i] > 25:
                position = -1
                signals[i] = -0.25
    
    return signals