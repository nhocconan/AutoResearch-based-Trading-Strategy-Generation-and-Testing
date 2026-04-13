#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d KAMA trend + RSI(14) extremes + choppiness regime filter
    # Long when: KAMA rising AND RSI < 30 (oversold) AND CHOP > 61.8 (range) → mean reversion long
    # Short when: KAMA falling AND RSI > 70 (overbought) AND CHOP > 61.8 (range) → mean reversion short
    # Exit when: RSI crosses 50 (mean reversion complete) OR CHOP < 38.2 (trend regime → follow trend)
    # Uses discrete sizing (0.25) targeting 50-100 total trades over 4 years.
    # Works in bull (buy dips in range) and bear (sell rallies in range) via RSI extremes.
    # Choppiness filter avoids whipsaws in strong trends; KAMA adapts to volatility.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for HTF trend filter (avoid trading against weekly trend)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate KAMA(10) on close
    def calculate_kama(close, period=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close, n=period))
        volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(close) > 1 else 0
        # Vectorized volatility calculation
        volatility = np.array([np.sum(np.abs(np.diff(close[i:i+period]))) 
                              for i in range(len(close)-period+1)])
        volatility = np.concatenate([np.full(period-1, np.nan), volatility])
        er = np.where(volatility != 0, change / volatility, 0)
        # Smoothing constants
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        # KAMA calculation
        kama = np.full_like(close, np.nan, dtype=float)
        kama[period] = close[period]  # seed
        for i in range(period+1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama = calculate_kama(close, period=10, fast=2, slow=30)
    
    # Calculate RSI(14)
    def calculate_rsi(close, period=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.full_like(close, np.nan, dtype=float)
        avg_loss = np.full_like(close, np.nan, dtype=float)
        if len(close) > period:
            avg_gain[period] = np.mean(gain[:period])
            avg_loss[period] = np.mean(loss[:period])
            for i in range(period+1, len(close)):
                avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
                avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close, period=14)
    
    # Calculate Choppiness Index(14)
    def calculate_chop(high, low, close, period=14):
        tr1 = high - low
        tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
        tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = np.full_like(close, np.nan, dtype=float)
        for i in range(period, len(close)):
            atr[i] = np.mean(tr[i-period+1:i+1])
        highest_high = np.full_like(close, np.nan, dtype=float)
        lowest_low = np.full_like(close, np.nan, dtype=float)
        for i in range(period-1, len(close)):
            highest_high[i] = np.max(high[i-period+1:i+1])
            lowest_low[i] = np.min(low[i-period+1:i+1])
        chop = np.full_like(close, np.nan, dtype=float)
        for i in range(period-1, len(close)):
            if atr[i] > 0 and highest_high[i] != lowest_low[i]:
                chop[i] = 100 * np.log10(np.sum(tr[i-period+1:i+1]) / 
                                          (period * np.log10(highest_high[i] - lowest_low[i]))) / np.log10(period)
        return chop
    
    chop = calculate_chop(high, low, close, period=14)
    
    # Get weekly trend: price > weekly EMA20 = bullish trend
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    weekly_bullish = close > ema_20_1w_aligned  # price above weekly EMA20
    
    # Align HTF indicators to 1d timeframe (wait for completed 1w bar)
    kama_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), kama)
    rsi_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), rsi)
    chop_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low, 'close': close}), chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(ema_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # KAMA direction: rising/falling
        kama_rising = kama_aligned[i] > kama_aligned[i-1]
        kama_falling = kama_aligned[i] < kama_aligned[i-1]
        
        # RSI extremes
        rsi_oversold = rsi_aligned[i] < 30
        rsi_overbought = rsi_aligned[i] > 70
        rsi_exit = (rsi_aligned[i] > 50 and rsi_aligned[i-1] <= 50) or \
                   (rsi_aligned[i] < 50 and rsi_aligned[i-1] >= 50)  # RSI crosses 50
        
        # Chop regime: > 61.8 = range (mean revert), < 38.2 = trending (follow trend)
        chop_range = chop_aligned[i] > 61.8
        chop_trending = chop_aligned[i] < 38.2
        
        # Entry conditions: KAMA direction + RSI extreme + chop range
        long_entry = (kama_rising and rsi_oversold and chop_range and 
                     weekly_bullish[i] and position != 1)
        short_entry = (kama_falling and rsi_overbought and chop_range and 
                      not weekly_bullish[i] and position != -1)
        
        # Exit conditions: RSI mean reversion complete OR chop becomes trending
        exit_long = (position == 1 and (rsi_exit or chop_trending))
        exit_short = (position == -1 and (rsi_exit or chop_trending))
        
        # Execute signals
        if long_entry:
            position = 1
            signals[i] = position_size
        elif short_entry:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_kama_rsi_chop_regime_v1"
timeframe = "1d"
leverage = 1.0