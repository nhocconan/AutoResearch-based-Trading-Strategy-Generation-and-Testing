#!/usr/bin/env python3
"""
1d_1day_MarketRegime_ADX_KAMA
Hypothesis: KAMA trend combined with ADX trend strength and range filter (Choppiness Index) filters whipsaws in choppy markets while capturing strong trends in both bull and bear markets. ADX > 25 indicates trending market, Choppiness > 61.8 indicates ranging market. Uses weekly trend filter for higher timeframe confirmation. Target: 10-25 trades/year (40-100 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # KAMA trend indicator
    def calculate_kama(price, er_length=10, fast=2, slow=30):
        change = np.abs(np.diff(price, prepend=price[0]))
        volatility = np.abs(np.diff(price, prepend=price[0]))
        er = np.zeros_like(price)
        er[er_length:] = np.abs(np.diff(price, n=er_length))[er_length-1:] / (
            np.convolve(volatility, np.ones(er_length), mode='same')[er_length-1:] + 1e-10)
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.zeros_like(price)
        kama[0] = price[0]
        for i in range(1, len(price)):
            kama[i] = kama[i-1] + sc[i] * (price[i] - kama[i-1])
        return kama
    
    # Choppiness Index
    def calculate_choppiness(high, low, close, period=14):
        atr = np.zeros_like(close)
        tr1 = np.abs(high - low)
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        atr = np.convolve(tr, np.ones(period)/period, mode='same')
        atr[:period-1] = np.nan
        
        highest_high = np.convolve(high, np.ones(period)/period, mode='same')
        lowest_low = np.convolve(low, np.ones(period)/period, mode='same')
        highest_high[:period-1] = np.nan
        lowest_low[:period-1] = np.nan
        
        chop = 100 * np.log10(atr.sum() / (np.nansum(highest_high - lowest_low) + 1e-10)) / np.log10(period)
        return chop
    
    # Calculate indicators
    kama = calculate_kama(close, er_length=10, fast=2, slow=30)
    
    # ADX calculation
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0) if high[i] - high[i-1] > low[i-1] - low[i] else 0
            minus_dm[i] = max(low[i-1] - low[i], 0) if low[i-1] - low[i] > high[i] - high[i-1] else 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        atr = np.convolve(tr, np.ones(period)/period, mode='same')
        atr[:period-1] = np.nan
        
        plus_di = 100 * np.convolve(plus_dm, np.ones(period)/period, mode='same') / (atr + 1e-10)
        minus_di = 100 * np.convolve(minus_dm, np.ones(period)/period, mode='same') / (atr + 1e-10)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = np.convolve(dx, np.ones(period)/period, mode='same')
        adx[:2*period-2] = np.nan
        return adx
    
    adx = calculate_adx(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly EMA 34 for trend direction
    ema_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or np.isnan(adx[i]) or np.isnan(chop[i]) or 
            np.isnan(ema_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama_val = kama[i]
        adx_val = adx[i]
        chop_val = chop[i]
        weekly_trend = ema_1w_aligned[i]
        
        # Only trade in trending markets (ADX > 25) and not in strong chop (Choppiness < 61.8)
        is_trending = adx_val > 25
        not_too_choppy = chop_val < 61.8
        
        if position == 0:
            # Long: price above KAMA and above weekly trend in trending market
            if price > kama_val and price > weekly_trend and is_trending and not_too_choppy:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA and below weekly trend in trending market
            elif price < kama_val and price < weekly_trend and is_trending and not_too_choppy:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Maintain long until price crosses below KAMA or weekly trend
            if price < kama_val or price < weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Maintain short until price crosses above KAMA or weekly trend
            if price > kama_val or price > weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1day_MarketRegime_ADX_KAMA"
timeframe = "1d"
leverage = 1.0