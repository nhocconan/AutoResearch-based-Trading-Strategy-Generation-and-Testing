#!/usr/bin/env python3
# 4H_Camarilla_Pullback_Trend
# Hypothesis: Buy pullbacks to Camarilla pivot levels during strong trends with volume confirmation.
# Uses 4h Camarilla levels calculated from previous day's OHLC.
# Long when: price pulls back to Camarilla L3/S3 level during uptrend (ADX>25) with volume > 1.5x average.
# Short when: price pulls back to Camarilla H3/H4 level during downtrend (ADX>25) with volume > 1.5x average.
# Uses 1d trend filter: only trade in direction of daily EMA50 trend.
# Works in bull/bear by following trend and using volume to confirm institutional interest.
# Target: 20-40 trades/year per symbol.

name = "4H_Camarilla_Pullback_Trend"
timeframe = "4h"
leverage = 1.0

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
    
    # 4h indicators
    close_s = pd.Series(close)
    volume_s = pd.Series(volume)
    
    # ADX for trend strength (14-period)
    # +DM and -DM
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Wilder's smoothing
    atr = np.zeros_like(tr)
    if len(tr) > 0:
        atr[0] = tr[0]
        for i in range(1, len(tr)):
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    # +DI and -DI
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean() / 
                     pd.Series(atr).ewm(alpha=1/14, adjust=False).mean())
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean() / 
                      pd.Series(atr).ewm(alpha=1/14, adjust=False).mean())
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    # Prepend NaN for alignment (since we lost first bar in calculations)
    adx = np.concatenate([np.full(1, np.nan), adx])
    
    # Volume average (20-period)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Calculate Camarilla levels from previous day's OHLC
    # Need daily OHLC for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla formula: 
    # H4 = close + 1.5 * (high - low)
    # H3 = close + 1.1 * (high - low)
    # L3 = close - 1.1 * (high - low)
    # L4 = close - 1.5 * (high - low)
    high_low = prev_high - prev_low
    H3 = prev_close + 1.1 * high_low
    L3 = prev_close - 1.1 * high_low
    
    # Align Camarilla levels to 4h (use previous day's levels)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # Daily trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    daily_uptrend = close_1d > ema50_1d
    daily_downtrend = close_1d < ema50_1d
    
    # Align daily trend to 4h
    daily_uptrend_aligned = align_htf_to_ltf(prices, df_1d, daily_uptrend.astype(float))
    daily_downtrend_aligned = align_htf_to_ltf(prices, df_1d, daily_downtrend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx[i]) or np.isnan(vol_ma[i]) or
            np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or
            np.isnan(daily_uptrend_aligned[i]) or np.isnan(daily_downtrend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        strong_trend = adx[i] > 25
        volume_confirm = vol_ratio > 1.5
        
        daily_up = daily_uptrend_aligned[i] > 0.5
        daily_down = daily_downtrend_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: daily uptrend + strong 4h trend + price near L3 + volume
            if daily_up and strong_trend and volume_confirm:
                if close[i] >= L3_aligned[i] * 0.995 and close[i] <= L3_aligned[i] * 1.005:
                    signals[i] = 0.25
                    position = 1
            # Enter short: daily downtrend + strong 4h trend + price near H3 + volume
            elif daily_down and strong_trend and volume_confirm:
                if close[i] >= H3_aligned[i] * 0.995 and close[i] <= H3_aligned[i] * 1.005:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit conditions: trend weakens or price moves away from L3
            if not daily_up or not strong_trend or close[i] < L3_aligned[i] * 0.98:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions: trend weakens or price moves away from H3
            if not daily_down or not strong_trend or close[i] > H3_aligned[i] * 1.02:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals