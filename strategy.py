#!/usr/bin/env python3
"""
1h ADX(14) + RSI(14) + 4h Trend Filter
Hypothesis: ADX filters trending markets, RSI captures pullbacks within trend, 4h EMA provides higher-timeframe direction.
Trades only during active London/NY session (08-20 UTC) to avoid low-volume periods. Target 60-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_adx_rsi_4htrend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC)
    hours = prices.index.hour
    
    # 14-period ATR for stops
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        atr[0] = np.nan
        if len(tr) > 0:
            atr[1] = tr[0]
            for i in range(2, n):
                atr[i] = (tr[i-1] * 13 + atr[i-1]) / 14
    
    # 14-period ADX
    adx = np.full(n, np.nan)
    if n >= 14:
        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)
        for i in range(1, n):
            up_move = high[i] - high[i-1]
            down_move = low[i-1] - low[i]
            plus_dm[i] = up_move if up_move > down_move and up_move > 0 else 0
            minus_dm[i] = down_move if down_move > up_move and down_move > 0 else 0
        
        tr_14 = np.zeros(n)
        tr_14[0] = np.nan
        for i in range(1, n):
            tr_14[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
        
        # Wilder's smoothing
        atr_14 = np.full(n, np.nan)
        plus_dm_14 = np.full(n, np.nan)
        minus_dm_14 = np.full(n, np.nan)
        if n >= 14:
            atr_14[13] = np.nanmean(tr_14[1:14])
            plus_dm_14[13] = np.nansum(plus_dm[1:14])
            minus_dm_14[13] = np.nansum(minus_dm[1:14])
            for i in range(14, n):
                atr_14[i] = (atr_14[i-1] * 13 + tr_14[i]) / 14
                plus_dm_14[i] = (plus_dm_14[i-1] * 13 + plus_dm[i]) / 14
                minus_dm_14[i] = (minus_dm_14[i-1] * 13 + minus_dm[i]) / 14
        
        plus_di = np.full(n, np.nan)
        minus_di = np.full(n, np.nan)
        dx = np.full(n, np.nan)
        for i in range(14, n):
            if atr_14[i] > 0:
                plus_di[i] = 100 * plus_dm_14[i] / atr_14[i]
                minus_di[i] = 100 * minus_dm_14[i] / atr_14[i]
                if plus_di[i] + minus_di[i] > 0:
                    dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        # ADX is smoothed DX
        adx[14] = np.nanmean(dx[15:29]) if n >= 29 else np.nan
        for i in range(29, n):
            adx[i] = (adx[i-1] * 13 + dx[i]) / 14
    
    # 14-period RSI
    rsi = np.full(n, np.nan)
    if n >= 14:
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full(n, np.nan)
        avg_loss = np.full(n, np.nan)
        if n >= 14:
            avg_gain[13] = np.mean(gain[1:14])
            avg_loss[13] = np.mean(loss[1:14])
            for i in range(14, n):
                avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
                avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
        
        rs = np.full(n, np.nan)
        for i in range(14, n):
            if avg_loss[i] > 0:
                rs[i] = avg_gain[i] / avg_loss[i]
                rsi[i] = 100 - (100 / (1 + rs[i]))
            else:
                rsi[i] = 100.0
    
    # 4h EMA(21) for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) >= 21:
        close_4h = df_4h['close'].values
        ema_4h = pd.Series(close_4h).ewm(span=21, adjust=False).mean().values
        ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    else:
        ema_4h_aligned = np.full(n, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start = max(30, 14)  # Warmup for indicators
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(adx[i]) or np.isnan(rsi[i]) or np.isnan(ema_4h_aligned[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: RSI > 70 (overbought) or ADX < 20 (trend weak)
            # Stoploss: price drops 2*ATR below entry
            if (rsi[i] > 70 or adx[i] < 20 or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: RSI < 30 (oversold) or ADX < 20 (trend weak)
            # Stoploss: price rises 2*ATR above entry
            if (rsi[i] < 30 or adx[i] < 20 or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries: RSI pullback in 4h trend direction
            # Long: 4h uptrend + RSI < 40 (pullback) + ADX > 25 (trending)
            # Short: 4h downtrend + RSI > 60 (pullback) + ADX > 25 (trending)
            if (ema_4h_aligned[i] > close[i] and  # 4h uptrend (price below EMA)
                rsi[i] < 40 and
                adx[i] > 25):
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            elif (ema_4h_aligned[i] < close[i] and  # 4h downtrend (price above EMA)
                  rsi[i] > 60 and
                  adx[i] > 25):
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals