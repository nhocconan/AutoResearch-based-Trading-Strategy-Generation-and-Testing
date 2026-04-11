#!/usr/bin/env python3
# 1d_1w_kama_rsi_chop_v1
# Strategy: 1d KAMA direction + RSI + Choppiness regime filter
# Timeframe: 1d
# Leverage: 1.0
# Hypothesis: KAMA adapts to market efficiency, providing smooth trend signals.
# RSI filters for overbought/oversold conditions in trending markets.
# Choppiness index (CHOP) identifies ranging vs trending regimes:
#   CHOP > 61.8 = ranging (favor mean reversion: RSI < 40 long, RSI > 60 short)
#   CHOP < 38.2 = trending (favor trend following: RSI > 50 long, RSI < 50 short)
# Works in both bull (trending) and bear (ranging) markets by adapting logic.
# Low trade frequency expected (<25/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_kama_rsi_chop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # --- 1d Indicators ---
    # KAMA (Kaufman Adaptive Moving Average)
    def calculate_kama(close, period=10, fast=2, slow=30):
        change = np.abs(np.diff(close, n=period))
        volatility = np.sum(np.abs(np.diff(close)), axis=0)
        er = np.zeros_like(close)
        er[period:] = change[period-1:] / np.maximum(volatility[period-1:], 1e-10)
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        kama = np.zeros_like(close)
        kama[period] = close[period]
        for i in range(period+1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama = calculate_kama(close, period=10, fast=2, slow=30)
    
    # RSI (14)
    def calculate_rsi(close, period=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        for i in range(period+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close, period=14)
    
    # Choppiness Index (14)
    def calculate_chop(high, low, close, period=14):
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        
        atr_sum = np.zeros_like(close)
        for i in range(1, len(close)):
            atr_sum[i] = atr_sum[i-1] + tr[i]
        
        hh = np.zeros_like(close)
        ll = np.zeros_like(close)
        hh[0] = high[0]
        ll[0] = low[0]
        for i in range(1, len(close)):
            hh[i] = max(hh[i-1], high[i])
            ll[i] = min(ll[i-1], low[i])
        
        chop = np.zeros_like(close)
        for i in range(period, len(close)):
            sum_tr = atr_sum[i] - atr_sum[i-period]
            hhll = hh[i] - ll[i]
            if hhll != 0:
                chop[i] = 100 * np.log10(sum_tr / hhll) / np.log10(period)
            else:
                chop[i] = 50
        return chop
    
    chop = calculate_chop(high, low, close, period=14)
    
    # --- Weekly Indicators (for regime/context) ---
    close_1w = df_1w['close'].values
    # Weekly EMA20 for trend context
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or np.isnan(ema_20_1w_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine market regime using Choppiness Index
        ranging = chop[i] > 61.8
        trending = chop[i] < 38.2
        
        # Trend direction from KAMA and weekly EMA
        price_above_kama = close[i] > kama[i]
        price_below_kama = close[i] < kama[i]
        weekly_uptrend = close[i] > ema_20_1w_aligned[i]
        weekly_downtrend = close[i] < ema_20_1w_aligned[i]
        
        # Entry logic based on regime
        if ranging:
            # In ranging markets: mean reversion at RSI extremes
            if rsi[i] < 40 and price_above_kama and position != 1:
                # Long when oversold and price above KAMA (bullish bias in range)
                position = 1
                signals[i] = 0.25
            elif rsi[i] > 60 and price_below_kama and position != -1:
                # Short when overbought and price below KAMA (bearish bias in range)
                position = -1
                signals[i] = -0.25
        elif trending:
            # In trending markets: follow momentum with RSI filter
            if price_above_kama and rsi[i] > 50 and weekly_uptrend and position != 1:
                # Long in uptrend with bullish momentum
                position = 1
                signals[i] = 0.25
            elif price_below_kama and rsi[i] < 50 and weekly_downtrend and position != -1:
                # Short in downtrend with bearish momentum
                position = -1
                signals[i] = -0.25
        
        # Exit conditions
        if position == 1:
            # Exit long: RSI overbought or price crosses below KAMA
            if rsi[i] > 70 or close[i] < kama[i]:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit short: RSI oversold or price crosses above KAMA
            if rsi[i] < 30 or close[i] > kama[i]:
                position = 0
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals