#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Momentum with 4h Trend Filter and 1d Regime Filter
# Uses 1h RSI(14) and momentum for entry timing, 4h EMA(50) for trend direction,
# and 1d ADX(14) to filter ranging markets. Designed for 15-35 trades/year.
# Works in bull markets (momentum + trend) and bear markets (mean reversion in range).
name = "1h_Momentum_4hEMA50_1dADX_Filter"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Get 1d data for regime filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 4h EMA50 for trend
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1d ADX for regime detection
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(arr, period):
        res = np.full_like(arr, np.nan)
        if len(arr) >= period:
            res[period-1] = np.nansum(arr[:period])
            for i in range(period, len(arr)):
                res[i] = res[i-1] - (res[i-1] / period) + arr[i]
        return res
    atr_1d = wilder_smooth(tr, 14)
    dm_plus_smooth = wilder_smooth(dm_plus, 14)
    dm_minus_smooth = wilder_smooth(dm_minus, 14)
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / atr_1d
    di_minus = 100 * dm_minus_smooth / atr_1d
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = wilder_smooth(dx, 14)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 1h RSI(14) for momentum
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    # Wilder's smoothing for RSI
    avg_gain = np.full_like(gain, np.nan)
    avg_loss = np.full_like(loss, np.nan)
    if len(gain) >= 14:
        avg_gain[13] = np.mean(gain[:14])
        avg_loss[13] = np.mean(loss[:14])
        for i in range(14, len(gain)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    # Prepend NaN for first element
    rsi = np.concatenate([[np.nan], rsi])
    
    # Calculate 1h price change momentum (4-period ROC)
    roc = np.full_like(close, np.nan)
    for i in range(4, len(close)):
        roc[i] = (close[i] - close[i-4]) / close[i-4] * 100
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(rsi[i]) or np.isnan(roc[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_val = ema_50_4h_aligned[i]
        adx_val = adx_aligned[i]
        rsi_val = rsi[i]
        roc_val = roc[i]
        
        if position == 0:
            # Long: Uptrend (price > EMA50), strong momentum (ROC > 0), not overbought (RSI < 70), trending market (ADX > 20)
            if close_val > ema_val and roc_val > 0 and rsi_val < 70 and adx_val > 20:
                signals[i] = 0.20
                position = 1
            # Short: Downtrend (price < EMA50), weak momentum (ROC < 0), not oversold (RSI > 30), trending market (ADX > 20)
            elif close_val < ema_val and roc_val < 0 and rsi_val > 30 and adx_val > 20:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: Trend change (price < EMA50) or overbought (RSI > 70) or range market (ADX < 15)
            if close_val < ema_val or rsi_val > 70 or adx_val < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: Trend change (price > EMA50) or oversold (RSI < 30) or range market (ADX < 15)
            if close_val > ema_val or rsi_val < 30 or adx_val < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals