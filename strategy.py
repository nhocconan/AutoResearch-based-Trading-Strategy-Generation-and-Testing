#!/usr/bin/env python3
"""
1h_Spike_Reversal_RangeFilter
Hypothesis: In 1h timeframe, enter reversals after volatility spikes in ranging markets.
- Use 4h ADX < 25 to identify ranging regime (avoid trends)
- Use 1h ATR ratio (ATR(1)/ATR(14)) > 1.5 to detect volatility spikes
- Enter long when spike occurs near support (low < BB lower band) and close > open
- Enter short when spike occurs near resistance (high > BB upper band) and close < open
- Use 1d trend filter: require price > 1d EMA50 for longs, price < 1d EMA50 for shorts
- Target: 20-40 trades/year to minimize fee decay while capturing mean reversion after spikes
- Works in both bull/bear: volatility spikes occur in all regimes, ranging filter avoids trend losses
"""

import numpy as np
import pandas as pd
from mf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    open_ = prices['open'].values
    volume = prices['volume'].values
    
    # 4h ADX for ranging filter (<25 = range)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate ADX components
    plus_dm = np.zeros(len(high_4h))
    minus_dm = np.zeros(len(high_4h))
    tr = np.zeros(len(high_4h))
    
    for i in range(1, len(high_4h)):
        high_diff = high_4h[i] - high_4h[i-1]
        low_diff = low_4h[i-1] - low_4h[i]
        plus_dm[i] = max(high_diff, 0) if high_diff > low_diff else 0
        minus_dm[i] = max(low_diff, 0) if low_diff > high_diff else 0
        tr[i] = max(high_4h[i] - low_4h[i], abs(high_4h[i] - close_4h[i-1]), abs(low_4h[i] - close_4h[i-1]))
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    period = 14
    if len(tr) >= period:
        atr = wilder_smooth(tr, period)
        plus_di = 100 * wilder_smooth(plus_dm, period) / atr
        minus_di = 100 * wilder_smooth(minus_dm, period) / atr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = wilder_smooth(dx, period)
    else:
        adx = np.full(len(high_4h), np.nan)
    
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx)
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1h indicators
    # ATR for volatility spike detection
    atr_1 = np.zeros(n)
    atr_14 = np.zeros(n)
    tr_1h = np.zeros(n)
    
    for i in range(1, n):
        tr_1h[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Wilder smoothing for ATR
    if len(tr_1h) >= 1:
        atr_1[0] = tr_1h[0]
        for i in range(1, n):
            atr_1[i] = (atr_1[i-1] * 0 + tr_1h[i])  # Simple ATR(1) = current TR
    
    if len(tr_1h) >= 14:
        atr_14[13] = np.nansum(tr_1h[:14])
        for i in range(14, n):
            atr_14[i] = (atr_14[i-1] * 13 + tr_1h[i]) / 14
    
    atr_ratio = np.where(atr_14 > 0, atr_1 / atr_14, 0)
    
    # Bollinger Bands (20, 2)
    close_s = pd.Series(close)
    bb_mid = close_s.rolling(window=20, min_periods=20).mean().values
    bb_std = close_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(20, 14, 50)  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(adx_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(atr_ratio[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Ranging filter: ADX < 25
        if adx_aligned[i] >= 25:
            signals[i] = 0.0
            continue
        
        # Volatility spike: ATR ratio > 1.5
        vol_spike = atr_ratio[i] > 1.5
        
        price = close[i]
        op = open_[i]
        
        if position == 0:
            # Long: volatility spike near support, bullish close, uptrend filter
            if (vol_spike and
                low[i] <= bb_lower[i] and  # spike reaches/below lower BB
                close[i] > op and          # bullish close
                price > ema_50_1d_aligned[i]):  # uptrend filter
                signals[i] = 0.20
                position = 1
            # Short: volatility spike near resistance, bearish close, downtrend filter
            elif (vol_spike and
                  high[i] >= bb_upper[i] and  # spike reaches/above upper BB
                  close[i] < op and           # bearish close
                  price < ema_50_1d_aligned[i]):  # downtrend filter
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            signals[i] = 0.20
            # Exit: volatility subsides or mean reversion complete
            if (atr_ratio[i] < 1.2 or  # volatility normalized
                close[i] >= bb_mid[i]):  # price reached middle band
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.20
            # Exit: volatility subsides or mean reversion complete
            if (atr_ratio[i] < 1.2 or  # volatility normalized
                close[i] <= bb_mid[i]):  # price reached middle band
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Spike_Reversal_RangeFilter"
timeframe = "1h"
leverage = 1.0