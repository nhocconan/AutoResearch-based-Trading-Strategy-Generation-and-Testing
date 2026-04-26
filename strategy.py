#!/usr/bin/env python3
"""
6h_KAMA_Regime_ADXFilter_v1
Hypothesis: 6h KAMA with regime filter (choppiness) and ADX trend strength.
- KAMA adapts to market noise: fast in trends, slow in chop
- Choppiness Index (CHOP) > 61.8 = range (mean revert), < 38.2 = trending (trend follow)
- ADX > 25 confirms trend strength for breakout entries
- Long when: KAMA rising, price > KAMA, CHOP < 38.2 (trending), ADX > 25
- Short when: KAMA falling, price < KAMA, CHOP < 38.2 (trending), ADX > 25
- Exit when CHOP > 50 (choppy) or opposite signal
- Designed for 12-37 trades/year (50-150 total over 4 years) to minimize fee drag
- Works in bull/bear markets by adapting to regime and using weekly trend filter for bias
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load weekly data ONCE for trend bias
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # Weekly EMA50 for trend bias (faster than 200 for more signals)
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # KAMA parameters
    er_period = 10
    fast_ema = 2
    slow_ema = 30
    
    # Efficiency Ratio (ER)
    change = np.abs(np.diff(close, n=er_period))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    # Handle volatility calculation properly
    volatility_series = pd.Series(close).rolling(window=er_period, min_periods=er_period).apply(
        lambda x: np.sum(np.abs(np.diff(x))), raw=True
    ).values
    er = np.where(volatility_series > 0, change / volatility_series, 0)
    
    # Smoothing constant (SC)
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    
    # KAMA calculation
    kama = np.full_like(close, np.nan)
    kama[er_period] = close[er_period]  # Seed
    for i in range(er_period + 1, n):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Choppiness Index (CHOP) - 14 period
    atr_period = 14
    tr1 = np.abs(high - low)
    tr2 = np.abs(np.roll(high, 1) - low)
    tr3 = np.abs(np.roll(low, 1) - high)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Set first TR to high-low
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    hh = pd.Series(high).rolling(window=atr_period, min_periods=atr_period).max().values
    ll = pd.Series(low).rolling(window=atr_period, min_periods=atr_period).min().values
    
    # Avoid division by zero
    chop = np.where((hh - ll) > 0, 
                    100 * np.log10(atr * atr_period / (hh - ll)) / np.log10(atr_period),
                    50)
    
    # ADX calculation (14 period)
    # +DM and -DM
    up_move = np.diff(high)
    down_move = -np.diff(low)
    up_move = np.insert(up_move, 0, 0)
    down_move = np.insert(down_move, 0, 0)
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_dm_14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm_14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    # DI values
    plus_di = np.where(tr_14 > 0, 100 * plus_dm_14 / tr_14, 0)
    minus_di = np.where(tr_14 > 0, 100 * minus_dm_14 / tr_14, 0)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) > 0, 
                  100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 
                  0)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align HTF weekly trend
    weekly_uptrend = close > ema50_1w_aligned
    weekly_downtrend = close < ema50_1w_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all indicators)
    start_idx = max(er_period, atr_period) + 14
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or np.isnan(chop[i]) or np.isnan(adx[i]) or
            np.isnan(ema50_1w_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # KAMA trend
        kama_rising = kama[i] > kama[i-1]
        kama_falling = kama[i] < kama[i-1]
        
        # Price vs KAMA
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        
        # Regime filters
        trending_market = chop[i] < 38.2
        choppy_market = chop[i] > 61.8
        strong_trend = adx[i] > 25
        
        if position == 0:
            # Long: KAMA rising, price > KAMA, trending market, strong trend, weekly uptrend bias
            if kama_rising and price_above_kama and trending_market and strong_trend and weekly_uptrend:
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling, price < KAMA, trending market, strong trend, weekly downtrend bias
            elif kama_falling and price_below_kama and trending_market and strong_trend and weekly_downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: choppy market OR KAMA turns down OR weekly trend turns down
            if choppy_market or not kama_rising or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: choppy market OR KAMA turns up OR weekly trend turns up
            if choppy_market or not kama_falling or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_KAMA_Regime_ADXFilter_v1"
timeframe = "6h"
leverage = 1.0