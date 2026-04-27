#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d ADX-based trend following with 1w EMA200 trend filter and volume confirmation.
# Long when ADX(14) > 25 (trending) and +DI(14) > -DI(14) (bullish momentum) with 1w EMA200 uptrend and volume > 1.3x average.
# Short when ADX(14) > 25 (trending) and -DI(14) > +DI(14) (bearish momentum) with 1w EMA200 downtrend and volume > 1.3x average.
# Exit when ADX(14) < 20 (weak trend) or DI crossover reverses.
# Uses ADX for robust trend strength and direction, targeting 12-25 trades per year on 1d timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA200 for trend filter
    ema_period = 200
    ema_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= ema_period:
        ema_1w[ema_period - 1] = np.mean(close_1w[:ema_period])
        for i in range(ema_period, len(close_1w)):
            ema_1w[i] = (close_1w[i] * (2 / (ema_period + 1)) + 
                         ema_1w[i - 1] * (1 - (2 / (ema_period + 1))))
    
    # Calculate ADX, +DI, -DI (14-period)
    adx_period = 14
    tr = np.maximum.reduce([
        high[1:] - low[1:],
        np.abs(high[1:] - close[:-1]),
        np.abs(low[1:] - close[:-1])
    ])
    tr = np.insert(tr, 0, high[0] - low[0])
    
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    plus_dm = np.insert(plus_dm, 0, 0)
    minus_dm = np.insert(minus_dm, 0, 0)
    
    atr = np.full(n, np.nan)
    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)
    dx = np.full(n, np.nan)
    adx = np.full(n, np.nan)
    
    # Initialize first ATR as first TR
    if n > 0:
        atr[adx_period - 1] = np.mean(tr[:adx_period])
        plus_di[adx_period - 1] = 100 * np.mean(plus_dm[:adx_period]) / atr[adx_period - 1] if atr[adx_period - 1] != 0 else 0
        minus_di[adx_period - 1] = 100 * np.mean(minus_dm[:adx_period]) / atr[adx_period - 1] if atr[adx_period - 1] != 0 else 0
        dx[adx_period - 1] = 100 * np.abs(plus_di[adx_period - 1] - minus_di[adx_period - 1]) / (plus_di[adx_period - 1] + minus_di[adx_period - 1]) if (plus_di[adx_period - 1] + minus_di[adx_period - 1]) != 0 else 0
    
    for i in range(adx_period, n):
        atr[i] = (atr[i-1] * (adx_period - 1) + tr[i]) / adx_period
        plus_di[i] = 100 * (plus_dm[i] + plus_di[i-1] * (adx_period - 1)) / (atr[i] * adx_period) if atr[i] != 0 else 0
        minus_di[i] = 100 * (minus_dm[i] + minus_di[i-1] * (adx_period - 1)) / (atr[i] * adx_period) if atr[i] != 0 else 0
        dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i]) if (plus_di[i] + minus_di[i]) != 0 else 0
        if i >= 2 * adx_period - 1:
            adx[i] = (adx[i-1] * (adx_period - 1) + dx[i]) / adx_period
        else:
            adx[i] = np.mean(dx[adx_period:i+1]) if not np.any(np.isnan(dx[adx_period:i+1])) else np.nan
    
    # Align 1w EMA200 to 1d timeframe
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume MA for confirmation (20-period)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i - 19:i + 1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need ADX, DI, EMA200, and volume MA20
    start_idx = max(2 * adx_period - 1, ema_period - 1, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(adx[i]) or np.isnan(plus_di[i]) or np.isnan(minus_di[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: require volume above average
        vol_filter = vol_now > 1.3 * vol_avg
        
        if position == 0:
            # Long: ADX > 25, +DI > -DI, 1w EMA200 uptrend, and volume filter
            if (adx[i] > 25 and plus_di[i] > minus_di[i] and 
                price > ema_1w_aligned[i] and vol_filter):
                signals[i] = size
                position = 1
            # Short: ADX > 25, -DI > +DI, 1w EMA200 downtrend, and volume filter
            elif (adx[i] > 25 and minus_di[i] > plus_di[i] and 
                  price < ema_1w_aligned[i] and vol_filter):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: ADX < 20 (weak trend) or -DI > +DI (bearish crossover)
            if adx[i] < 20 or minus_di[i] > plus_di[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: ADX < 20 (weak trend) or +DI > -DI (bullish crossover)
            if adx[i] < 20 or plus_di[i] > minus_di[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_ADX14_Trend_1wEMA200_Volume"
timeframe = "1d"
leverage = 1.0