#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h timeframe with 4h ADX trend filter and 1d RSI mean reversion
# Uses 4h ADX > 25 to identify trending markets, then 1h momentum (price > EMA20) for entries
# In ranging markets (ADX <= 25), uses 1d RSI extremes (<30 or >70) for mean reversion
# Volume confirmation requires > 1.5x 20-bar median volume
# Conservative sizing (0.20) to limit trade frequency and avoid fee drag
# Session filter: 08-20 UTC to avoid low-volume Asian session

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    # 4h ADX for trend strength
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # True Range
    tr1 = np.abs(high_4h[1:] - low_4h[1:])
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_4h[1:] - high_4h[:-1]) > (low_4h[:-1] - low_4h[1:]), 
                       np.maximum(high_4h[1:] - high_4h[:-1], 0), 0)
    dm_minus = np.where((low_4h[:-1] - low_4h[1:]) > (high_4h[1:] - high_4h[:-1]), 
                        np.maximum(low_4h[:-1] - low_4h[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth TR, DM+, DM- with Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nansum(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr = wilder_smooth(tr, 14)
    dm_plus_smooth = wilder_smooth(dm_plus, 14)
    dm_minus_smooth = wilder_smooth(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr > 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr > 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilder_smooth(dx, 14)
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx)
    
    # 1h EMA20 for momentum
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # 1d RSI for mean reversion
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d.values)
    
    # Volume confirmation: current > 1.5x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(30, n):  # Start after warmup
        # Skip if any required data is NaN or outside session
        if (np.isnan(adx_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(vol_threshold[i]) or not session_mask[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        
        # Trending market: ADX > 25
        if adx_aligned[i] > 25:
            # Long: price > EMA20 (uptrend momentum)
            if close[i] > ema20[i] and volume[i] > vol_threshold[i]:
                signals[i] = 0.20
            # Short: price < EMA20 (downtrend momentum)
            elif close[i] < ema20[i] and volume[i] > vol_threshold[i]:
                signals[i] = -0.20
            else:
                signals[i] = signals[i-1] if i > 0 else 0.0
        # Ranging market: ADX <= 25
        else:
            # Long: RSI oversold (<30)
            if rsi_1d_aligned[i] < 30 and volume[i] > vol_threshold[i]:
                signals[i] = 0.20
            # Short: RSI overbought (>70)
            elif rsi_1d_aligned[i] > 70 and volume[i] > vol_threshold[i]:
                signals[i] = -0.20
            else:
                signals[i] = signals[i-1] if i > 0 else 0.0
    
    return signals

name = "1h_ADX_Trend_RSI_MeanReversion"
timeframe = "1h"
leverage = 1.0