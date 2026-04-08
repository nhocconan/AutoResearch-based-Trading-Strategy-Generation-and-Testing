#!/usr/bin/env python3
# 1h_4h_1d_trend_follow_volume_v2
# Hypothesis: Follow the 4h trend using EMA21/50 cross and ADX > 25, with 1d regime filter (price > SMA50).
# Enter on 1h breakouts above/below 4h EMA21 with volume confirmation. Works in bull/bear by aligning with higher timeframe trend.
# Uses volume spike to confirm institutional participation and reduce false signals.
# Target: 60-150 total trades over 4 years (15-37/year).

name = "1h_4h_1d_trend_follow_volume_v2"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h trend: EMA crossover
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate EMA21 and EMA50 on 4h close
    close_4h = df_4h['close'].values
    ema21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 4h trend direction: 1 if EMA21 > EMA50, -1 if EMA21 < EMA50
    trend_4h = np.where(ema21_4h > ema50_4h, 1, -1)
    
    # 4h ADX for trend strength
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            if plus_dm[i] == minus_dm[i]:
                plus_dm[i] = 0
                minus_dm[i] = 0
            elif plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            else:
                minus_dm[i] = 0
            
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
        
        # Wilder's smoothing
        atr = np.zeros_like(high)
        atr[period-1] = np.mean(tr[1:period]) if period < len(tr) else np.nan
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values / atr
        minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values / atr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = pd.Series(dx).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
        
        return adx
    
    adx_4h = calculate_adx(df_4h['high'].values, df_4h['low'].values, df_4h['close'].values)
    
    # Align 4h indicators to 1h
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    
    # 1d regime filter: avoid ranging markets
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    sma50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    
    # Market regime: 1 if trending (price > SMA50), -1 if ranging (price < SMA50)
    regime_1d = np.where(close_1d > sma50_1d, 1, -1)
    regime_1d_aligned = align_htf_to_ltf(prices, df_1d, regime_1d)
    
    # 4h EMA21 aligned to 1h for entry timing
    ema21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema21_4h)
    
    # Volume spike detection on 1h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_spike = vol_ratio > 1.5  # 50% above average volume
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(50, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(trend_4h_aligned[i]) or np.isnan(adx_4h_aligned[i]) or 
            np.isnan(regime_1d_aligned[i]) or np.isnan(ema21_4h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Only trade in session and with volume spike
        if not (in_session[i] and vol_spike[i]):
            if position != 0:
                # Hold position until exit signal
                pass
            else:
                signals[i] = 0.0
                continue
        
        if position == 1:  # Long position
            # Exit: 4h trend turns bearish OR ADX weakens
            if trend_4h_aligned[i] == -1 or adx_4h_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: 4h trend turns bullish OR ADX weakens
            if trend_4h_aligned[i] == 1 or adx_4h_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Need strong trend (ADX > 25) and favorable regime
            if adx_4h_aligned[i] > 25 and regime_1d_aligned[i] == trend_4h_aligned[i]:
                # Long: 4h bullish trend + price above 4h EMA21 (breakout entry)
                if trend_4h_aligned[i] == 1 and close[i] > ema21_4h_aligned[i]:
                    position = 1
                    signals[i] = 0.20
                # Short: 4h bearish trend + price below 4h EMA21
                elif trend_4h_aligned[i] == -1 and close[i] < ema21_4h_aligned[i]:
                    position = -1
                    signals[i] = -0.20
    
    return signals