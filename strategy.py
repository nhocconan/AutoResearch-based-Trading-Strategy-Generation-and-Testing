#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h timeframe with 4h trend filter and 1d regime filter to reduce noise.
# Uses 4h EMA for trend direction, 1d ADX for trend strength, and 1h RSI for entry timing.
# Only trades during 08-20 UTC to avoid low-liquidity hours.
# Position size: 0.20 (20% of capital) to limit drawdown.
# Target: 15-30 trades/year (~60-120 total over 4 years) to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # 4h EMA for trend direction
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1d ADX for trend strength (needs extra delay for Welles Wilder's smoothing)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values (Wilder's smoothing: alpha = 1/period)
    def wildeer_smooth(x, period):
        smoothed = np.full_like(x, np.nan, dtype=float)
        smoothed[period-1] = np.nansum(x[:period])
        for i in range(period, len(x)):
            smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + x[i]
        return smoothed
    
    atr = wildeer_smooth(tr, 14)
    dm_plus_smooth = wildeer_smooth(dm_plus, 14)
    dm_minus_smooth = wildeer_smooth(dm_minus, 14)
    
    # DI and DX
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = wildeer_smooth(dx, 14)
    
    # Align ADX with extra delay (ADX needs confirmation)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx, additional_delay_bars=2)
    
    # 1h RSI for entry timing
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    def rsi_wilder(gain, loss, period=14):
        avg_gain = np.full_like(gain, np.nan, dtype=float)
        avg_loss = np.full_like(loss, np.nan, dtype=float)
        avg_gain[period-1] = np.mean(gain[1:period+1]) if period < len(gain) else np.nan
        avg_loss[period-1] = np.mean(loss[1:period+1]) if period < len(loss) else np.nan
        for i in range(period, len(gain)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        return 100 - (100 / (1 + rs))
    
    rsi = rsi_wilder(gain, loss, 14)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Generate signals
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # warmup period
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any NaN values
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(close[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 4h EMA
        uptrend = close[i] > ema_4h_aligned[i]
        downtrend = close[i] < ema_4h_aligned[i]
        
        # Trend strength filter: ADX > 25
        strong_trend = adx_aligned[i] > 25
        
        if position == 0:
            # Long: uptrend, strong trend, RSI > 55 (momentum)
            if uptrend and strong_trend and rsi[i] > 55:
                signals[i] = 0.20
                position = 1
            # Short: downtrend, strong trend, RSI < 45 (momentum)
            elif downtrend and strong_trend and rsi[i] < 45:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: trend weakens or RSI overbought
            if not (uptrend and strong_trend) or rsi[i] > 75:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: trend weakens or RSI oversold
            if not (downtrend and strong_trend) or rsi[i] < 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4hEMA_1dADX_RSI_Session"
timeframe = "1h"
leverage = 1.0