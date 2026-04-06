#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h momentum with 4h trend filter and volume confirmation.
# Uses 4h ADX(14) for trend strength, 1h RSI(14) for momentum, volume filter to avoid false signals.
# Trades only during 08-20 UTC session to avoid low-liquidity periods.
# Targets 15-30 trades/year (60-120 over 4 years) to minimize fee drag.
# Works in bull/bear by requiring strong trend (ADX>25) and momentum alignment.

name = "1h_adx14_rsi14_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 14-period ATR for stops
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[14] = np.mean(tr[:14])
            for i in range(15, n):
                atr[i] = (atr[i-1] * 13 + tr[i-1]) / 14
    
    # 14-period RSI on 1h
    rsi = np.full(n, np.nan)
    if n >= 15:
        delta = np.diff(close)
        up = np.where(delta > 0, delta, 0)
        down = np.where(delta < 0, -delta, 0)
        roll_up = pd.Series(up).ewm(alpha=1/14, adjust=False).mean()
        roll_down = pd.Series(down).ewm(alpha=1/14, adjust=False).mean()
        rs = roll_up / roll_down.replace(0, np.nan)
        rsi[14:] = 100 - (100 / (1 + rs[14:].values))
    
    # 14-period ADX on 4h
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate True Range and Directional Movement for ADX
    tr_4h = np.maximum(
        high_4h[1:] - low_4h[1:],
        np.abs(high_4h[1:] - close_4h[:-1]),
        np.abs(low_4h[1:] - close_4h[:-1])
    )
    plus_dm = np.where((high_4h[1:] - high_4h[:-1]) > (low_4h[:-1] - low_4h[1:]), 
                       np.maximum(high_4h[1:] - high_4h[:-1], 0), 0)
    minus_dm = np.where((low_4h[:-1] - low_4h[1:]) > (high_4h[1:] - high_4h[:-1]), 
                        np.maximum(low_4h[:-1] - low_4h[1:], 0), 0)
    
    # Smooth TR, +DM, -DM over 14 periods
    tr_14 = np.full(len(tr_4h), np.nan)
    plus_dm_14 = np.full(len(plus_dm), np.nan)
    minus_dm_14 = np.full(len(minus_dm), np.nan)
    
    if len(tr_4h) >= 14:
        tr_14[13] = np.mean(tr_4h[:14])
        plus_dm_14[13] = np.mean(plus_dm[:14])
        minus_dm_14[13] = np.mean(minus_dm[:14])
        for i in range(14, len(tr_4h)):
            tr_14[i] = (tr_14[i-1] * 13 + tr_4h[i]) / 14
            plus_dm_14[i] = (plus_dm_14[i-1] * 13 + plus_dm[i]) / 14
            minus_dm_14[i] = (minus_dm_14[i-1] * 13 + minus_dm[i]) / 14
    
    # Calculate +DI and -DI
    plus_di = np.full(len(tr_14), np.nan)
    minus_di = np.full(len(tr_14), np.nan)
    dx = np.full(len(tr_14), np.nan)
    
    valid = ~np.isnan(tr_14) & (tr_14 != 0)
    if np.any(valid):
        plus_di[valid] = 100 * plus_dm_14[valid] / tr_14[valid]
        minus_di[valid] = 100 * minus_dm_14[valid] / tr_14[valid]
        dx[valid] = 100 * np.abs(plus_di[valid] - minus_di[valid]) / (plus_di[valid] + minus_di[valid])
    
    # Calculate ADX (smoothed DX)
    adx_4h = np.full(len(dx), np.nan)
    if len(dx) >= 14:
        valid_dx = ~np.isnan(dx)
        if np.any(valid_dx):
            first_valid = np.where(valid_dx)[0][0]
            if first_valid + 13 < len(dx):
                adx_4h[first_valid + 13] = np.mean(dx[first_valid:first_valid + 14])
                for i in range(first_valid + 14, len(dx)):
                    if not np.isnan(dx[i]):
                        adx_4h[i] = (adx_4h[i-1] * 13 + dx[i]) / 14
    
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    
    # Volume filter: current volume > 1.3x average over last 20 periods
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(30, 15, 20)
    
    for i in range(start, n):
        # Skip if required data not available or outside session
        if (np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(vol_ma[i]) or hours[i] < 8 or hours[i] > 20):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.3
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: RSI < 40 or stoploss hit
            if (rsi[i] < 40 or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: RSI > 60 or stoploss hit
            if (rsi[i] > 60 or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries
            # Long: RSI > 50, ADX > 25 (strong trend), volume filter
            if (rsi[i] > 50 and adx_aligned[i] > 25 and volume_filter):
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            # Short: RSI < 50, ADX > 25 (strong trend), volume filter
            elif (rsi[i] < 50 and adx_aligned[i] > 25 and volume_filter):
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals