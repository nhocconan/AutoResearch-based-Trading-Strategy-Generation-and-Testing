# Hypothesis: 4h timeframe with volume-weighted price action (VWAP) deviation + trend filter (ADX) + volatility filter (ATR-based stop)
# VWAP deviation identifies mean-reversion opportunities in trending markets, with ADX ensuring sufficient trend strength.
# Uses 1d timeframe for VWAP anchor and trend context to avoid look-ahead and ensure proper alignment.
# Targets 20-40 trades/year to minimize fee drag while capturing meaningful moves.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for VWAP anchor and trend context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily VWAP using typical price and volume
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    vwap_1d = (typical_price_1d * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    vwap_1d = vwap_1d.values
    
    # Align daily VWAP to 4h
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # 4h ATR for volatility normalization and stop
    atr_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = np.nan
    
    atr = np.full(n, np.nan)
    if n >= atr_period:
        atr[atr_period-1] = np.nanmean(tr[:atr_period])
        for i in range(atr_period, n):
            atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # 1d ADX for trend strength filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    if len(high_1d) < 14:
        return np.zeros(n)
    
    # True Range
    tr1_1d = np.abs(high_1d - low_1d)
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    tr_1d[0] = np.nan
    
    # Directional Movement
    dm_plus_1d = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                          np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus_1d = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                           np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus_1d[0] = 0
    dm_minus_1d[0] = 0
    
    # Wilder's smoothing
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr_1d = wilder_smooth(tr_1d, 14)
    dm_plus_smooth = wilder_smooth(dm_plus_1d, 14)
    dm_minus_smooth = wilder_smooth(dm_minus_1d, 14)
    
    di_plus = 100 * dm_plus_smooth / np.where(atr_1d == 0, 1, atr_1d)
    di_minus = 100 * dm_minus_smooth / np.where(atr_1d == 0, 1, atr_1d)
    dx = np.where((di_plus + di_minus) == 0, 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus))
    adx_1d = wilder_smooth(dx, 14)
    
    # Align ADX to 4h
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need VWAP (1), ATR (14), ADX (14+14=28)
    start_idx = max(1, atr_period, 28)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(vwap_1d_aligned[i]) or np.isnan(atr[i]) or np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vwap = vwap_1d_aligned[i]
        atr_val = atr[i]
        adx = adx_1d_aligned[i]
        
        # Avoid division by zero
        if atr_val == 0:
            signals[i] = 0.0
            continue
        
        # Calculate deviation from VWAP in ATR units
        deviation = (price - vwap) / atr_val
        
        # Trend filter: only trade when ADX > 25 (trending market)
        trend_filter = adx > 25
        
        if position == 0:
            # Long: price deviates below VWAP (mean reversion long) in uptrend
            if deviation < -0.5 and trend_filter:
                signals[i] = size
                position = 1
            # Short: price deviates above VWAP (mean reversion short) in downtrend
            elif deviation > 0.5 and trend_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to VWAP or trend weakens
            if deviation > -0.2 or adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to VWAP or trend weakens
            if deviation < 0.2 or adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_VWAP_Deviation_ADX25_Trend"
timeframe = "4h"
leverage = 1.0