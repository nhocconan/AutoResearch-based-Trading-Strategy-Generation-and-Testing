#!/usr/bin/env python3
"""
1h_SMMA_RSI_Breakout_v1
Hypothesis: Combines Smoothed Moving Average (SMMA) trend with RSI momentum on 1h timeframe, filtered by 4h trend direction and volume confirmation.
Uses 4h EMA50 for trend filter and 1d ADX for trend strength to avoid false signals in choppy markets.
Designed for low trade frequency (~20-30 trades/year) by requiring confluence of SMMA crossover, RSI momentum, volume spike, and trend alignment.
Works in both bull and bear markets by adapting to trend direction via 4h EMA filter and only taking trades in strong trends (ADX > 25).
"""

name = "1h_SMMA_RSI_Breakout_v1"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 4h data for trend filter and 1d data for ADX trend strength
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 50 or len(df_1d) < 30:
        return np.zeros(n)
    
    # 1h OHLCV
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 4h EMA50 for trend filter ---
    close_4h = df_4h['close']
    ema_50_4h = close_4h.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # --- SMMA (Smoothed Moving Average) 20-period on 1h close ---
    # SMMA is similar to RMA/Wilder's smoothing: SMMA(t) = (SMMA(y-1) * (N-1) + X(t)) / N
    # We'll implement as EMA with alpha = 1/N for equivalence after warmup
    smma = np.zeros_like(close)
    smma[0] = close[0]
    for i in range(1, n):
        smma[i] = (smma[i-1] * 19 + close[i]) / 20  # 20-period SMMA
    
    # --- RSI (14-period) on 1h close ---
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    avg_gain[13] = np.mean(gain[1:14])  # First average of first 14 periods
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.where(avg_loss == 0, 100, rsi)  # Handle division by zero
    
    # --- Volume Spike Detection (1.8x 24-period EMA) ---
    vol_ema = pd.Series(volume).ewm(span=24, adjust=False, min_periods=24).mean().values
    vol_spike = volume > (1.8 * vol_ema)
    
    # --- ADX (14-period) on 1d for trend strength ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    
    # Smooth TR, DM+, DM- with Wilder's smoothing (14-period)
    def wilder_smooth(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.mean(data[1:period])  # First average
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr = wilder_smooth(tr, 14)
    dm_plus_smooth = wilder_smooth(dm_plus, 14)
    dm_minus_smooth = wilder_smooth(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / np.where(atr != 0, atr, 1e-10)
    di_minus = 100 * dm_minus_smooth / np.where(atr != 0, atr, 1e-10)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0,
                  100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilder_smooth(dx, 14)
    
    # Align ADX to 1h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (max of all indicators)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(smma[i]) or
            np.isnan(rsi[i]) or
            np.isnan(vol_spike[i]) or
            np.isnan(adx_aligned[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
            continue
        
        # Determine trend based on price vs 4h EMA50
        price_above_ema = close[i] > ema_50_4h_aligned[i]
        price_below_ema = close[i] < ema_50_4h_aligned[i]
        
        # SMMA signal: price above/below SMMA indicates momentum
        price_above_smma = close[i] > smma[i]
        price_below_smma = close[i] < smma[i]
        
        # RSI momentum: > 55 = bullish, < 45 = bearish (avoid chop around 50)
        rsi_bullish = rsi[i] > 55
        rsi_bearish = rsi[i] < 45
        
        # Trend strength filter: only trade in strong trends (ADX > 25)
        strong_trend = adx_aligned[i] > 25
        
        if position == 0:
            # Look for breakout in direction of trend with momentum confirmation
            if strong_trend and vol_spike[i]:
                if price_above_ema and price_above_smma and rsi_bullish:
                    signals[i] = 0.20
                    position = 1
                elif price_below_ema and price_below_smma and rsi_bearish:
                    signals[i] = -0.20
                    position = -1
        else:
            # Exit conditions: reverse signals or loss of momentum
            if position == 1:
                # Exit long: price breaks below SMMA or RSI loses momentum
                exit_signal = (price_below_smma) or (rsi[i] < 50)
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            elif position == -1:
                # Exit short: price breaks above SMMA or RSI loses momentum
                exit_signal = (price_above_smma) or (rsi[i] > 50)
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals