#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Adaptive_Kelly_Vol_Trend_Regime"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for regime and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily ATR for volatility regime
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    tr1 = daily_high - daily_low
    tr2 = np.abs(daily_high - np.roll(daily_close, 1))
    tr3 = np.abs(daily_low - np.roll(daily_close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_daily = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Daily trend: EMA50 vs EMA200
    ema50_daily = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_daily = pd.Series(daily_close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align daily indicators to 6h
    atr_daily_6h = align_htf_to_ltf(prices, df_1d, atr_daily)
    ema50_daily_6h = align_htf_to_ltf(prices, df_1d, ema50_daily)
    ema200_daily_6h = align_htf_to_ltf(prices, df_1d, ema200_daily)
    
    # Volatility ratio: short-term vs long-term ATR (volatility regime filter)
    atr_short = pd.Series(tr).rolling(window=7, min_periods=7).mean().values
    atr_long = pd.Series(tr).rolling(window=30, min_periods=30).mean().values
    atr_ratio = atr_short / atr_long
    atr_ratio_6h = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # 6h price momentum: ROC(6)
    roc_6h = np.zeros_like(close)
    roc_6h[6:] = (close[6:] - close[:-6]) / close[:-6] * 100
    
    # 6h RSI(14) for mean reversion signals
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: above 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(atr_daily_6h[i]) or np.isnan(ema50_daily_6h[i]) or 
            np.isnan(ema200_daily_6h[i]) or np.isnan(atr_ratio_6h[i]) or
            np.isnan(rsi[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filters
        vol_expanding = atr_ratio_6h[i] > 1.1  # Volatility expanding
        vol_contracting = atr_ratio_6h[i] < 0.9  # Volatility contracting
        daily_uptrend = ema50_daily_6h[i] > ema200_daily_6h[i]
        daily_downtrend = ema50_daily_6h[i] < ema200_daily_6h[i]
        
        vol_ok = volume[i] > 1.3 * vol_ma[i]
        
        # Session filter: 08-20 UTC
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        in_session = 8 <= hour <= 20
        
        if position == 0:
            # Long conditions: volatility contracting + uptrend + RSI oversold bounce
            if (vol_contracting and daily_uptrend and 
                rsi[i] < 35 and roc_6h[i] > 0 and 
                vol_ok and in_session):
                # Kelly fraction approximation based on recent win rate
                win_rate_est = 0.55  # conservative estimate
                kelly_f = max(0.1, min(0.3, 2 * win_rate_est - 1))  # simplified Kelly
                signals[i] = kelly_f
                position = 1
            # Short conditions: volatility contracting + downtrend + RSI overbought bounce
            elif (vol_contracting and daily_downtrend and 
                  rsi[i] > 65 and roc_6h[i] < 0 and 
                  vol_ok and in_session):
                win_rate_est = 0.55
                kelly_f = max(0.1, min(0.3, 2 * win_rate_est - 1))
                signals[i] = -kelly_f
                position = -1
        
        elif position == 1:
            # Exit long: volatility expanding OR RSI overbought
            if vol_expanding or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = kelly_f if 'kelly_f' in locals() else 0.25
        
        elif position == -1:
            # Exit short: volatility expanding OR RSI oversold
            if vol_expanding or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -kelly_f if 'kelly_f' in locals() else -0.25
    
    return signals