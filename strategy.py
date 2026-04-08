#!/usr/bin/env python3
# 1d_2025_scalp_v1
# Hypothesis: Follow the 1w trend using EMA crossover and ADX strength, with 1d regime filter to avoid sideways markets.
# Enter on 1d pullbacks in the direction of the 1w trend with volume confirmation. Works in bull/bear by aligning with higher timeframe trend.
# Uses volume spike to confirm institutional participation and reduce false signals.
# Target: 30-100 total trades over 4 years (7-25/year).

name = "1d_2025_scalp_v1"
timeframe = "1d"
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
    
    # 1w trend: EMA crossover
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate EMA21 and EMA50 on 1w close
    close_1w = df_1w['close'].values
    ema21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 1w trend direction: 1 if EMA21 > EMA50, -1 if EMA21 < EMA50
    trend_1w = np.where(ema21_1w > ema50_1w, 1, -1)
    
    # 1w ADX for trend strength
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
    
    adx_1w = calculate_adx(df_1w['high'].values, df_1w['low'].values, df_1w['close'].values)
    
    # Align 1w indicators to 1d
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # 1d regime filter: avoid ranging markets
    close_1d = prices['close'].values
    sma50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    
    # Market regime: 1 if trending (price > SMA50), -1 if ranging (price < SMA50)
    regime_1d = np.where(close_1d > sma50_1d, 1, -1)
    
    # Volume spike detection on 1d
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
        if (np.isnan(trend_1w_aligned[i]) or np.isnan(adx_1w_aligned[i]) or 
            np.isnan(sma50_1d[i]) or np.isnan(vol_ma[i])):
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
            # Exit: 1w trend turns bearish OR ADX weakens
            if trend_1w_aligned[i] == -1 or adx_1w_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: 1w trend turns bullish OR ADX weakens
            if trend_1w_aligned[i] == 1 or adx_1w_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Need strong trend (ADX > 25) and favorable regime
            if adx_1w_aligned[i] > 25 and regime_1d == trend_1w_aligned[i]:
                # Long: 1w bullish trend + price above 1d EMA21 (pullback entry)
                if trend_1w_aligned[i] == 1:
                    ema21_1d = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
                    if not np.isnan(ema21_1d[i]) and close[i] > ema21_1d[i]:
                        position = 1
                        signals[i] = 0.20
                # Short: 1w bearish trend + price below 1d EMA21
                elif trend_1w_aligned[i] == -1:
                    ema21_1d = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
                    if not np.isnan(ema21_1d[i]) and close[i] < ema21_1d[i]:
                        position = -1
                        signals[i] = -0.20
    
    return signals