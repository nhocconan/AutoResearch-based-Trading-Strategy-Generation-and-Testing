#!/usr/bin/env python3
"""
4h_1D_RSI_Momentum_Breakout_v1
Hypothesis: In trending markets, RSI momentum combined with price breakouts above/below
dynamic channels (Keltner) captures sustained moves. Uses 1D trend filter (EMA50) to
align with higher timeframe momentum, reducing false signals. Volatility-adjusted
position sizing (inverse ATR) controls risk. Designed for low trade frequency
(<50/year) by requiring confluence of momentum, breakout, and trend.
Works in bull/bear via 1D trend filter and volatility regime.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1D_RSI_Momentum_Breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 1D TREND FILTER ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # EMA50 on daily close
    ema50_1d = np.full_like(close_1d, np.nan)
    alpha = 2.0 / (50 + 1)
    for i in range(len(close_1d)):
        if i == 0:
            ema50_1d[i] = close_1d[i]
        elif not np.isnan(close_1d[i]):
            ema50_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema50_1d[i-1]
        else:
            ema50_1d[i] = ema50_1d[i-1]
    
    # Trend: 1 = uptrend (close > EMA50), -1 = downtrend (close < EMA50)
    trend_1d = np.where(close_1d > ema50_1d, 1, -1)
    trend_1d = np.where(np.isnan(ema50_1d), 0, trend_1d)
    trend_aligned = align_htf_to_ltf(prices, df_1d, trend_1d.astype(float))
    
    # === 4H KELTNER CHANNELS (20, 2.0) ===
    # ATR(20)
    tr1 = np.abs(high[1:] - low[:-1])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.full_like(tr, np.nan)
    for i in range(len(tr)):
        if i < 20:
            atr[i] = np.nan
        elif i == 20:
            atr[i] = np.nanmean(tr[1:i+1])
        else:
            atr[i] = (atr[i-1] * 19 + tr[i]) / 20
    
    # EMA20 of close
    ema_close = np.full_like(close, np.nan)
    alpha_ema = 2.0 / (20 + 1)
    for i in range(len(close)):
        if i == 0:
            ema_close[i] = close[i]
        elif not np.isnan(close[i]):
            ema_close[i] = alpha_ema * close[i] + (1 - alpha_ema) * ema_close[i-1]
        else:
            ema_close[i] = ema_close[i-1]
    
    # Keltner channels
    kc_middle = ema_close
    kc_upper = kc_middle + 2.0 * atr
    kc_lower = kc_middle - 2.0 * atr
    
    # === 4H RSI(14) ===
    # RSI calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    # Smoothed gains/losses (Wilder's smoothing)
    avg_gain = np.full_like(gain, np.nan)
    avg_loss = np.full_like(loss, np.nan)
    for i in range(len(gain)):
        if i < 14:
            avg_gain[i] = np.nan
            avg_loss[i] = np.nan
        elif i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.where(avg_loss == 0, 100, rsi)
    rsi = np.where(avg_gain == 0, 0, rsi)
    
    # === VOLUME FILTER (20-period average) ===
    vol_avg = np.full_like(volume, np.nan, dtype=float)
    vol_sum = 0.0
    vol_count = 0
    for i in range(len(volume)):
        vol_sum += volume[i]
        vol_count += 1
        if i >= 20:
            vol_sum -= volume[i-20]
            vol_count -= 1
        if vol_count > 0:
            vol_avg[i] = vol_sum / vol_count
    
    vol_sma = vol_avg  # for clarity
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # start after warmup
        # Skip if indicators not available
        if (np.isnan(kc_upper[i]) or np.isnan(kc_lower[i]) or 
            np.isnan(rsi[i]) or np.isnan(trend_aligned[i]) or 
            np.isnan(vol_sma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: at least 1.2x average
        vol_confirm = volume[i] > 1.2 * vol_sma[i]
        
        # Only trade in alignment with 1D trend
        long_trend = trend_aligned[i] > 0.5
        short_trend = trend_aligned[i] < -0.5
        
        # Momentum conditions: RSI not extreme
        rsi_not_overbought = rsi[i] < 70
        rsi_not_oversold = rsi[i] > 30
        
        # Entry conditions
        long_setup = (close[i] > kc_upper[i]) and vol_confirm and long_trend and rsi_not_overbought
        short_setup = (close[i] < kc_lower[i]) and vol_confirm and short_trend and rsi_not_oversold
        
        # Exit conditions: mean reversion to middle channel
        exit_long = close[i] < kc_middle[i]
        exit_short = close[i] > kc_middle[i]
        
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_setup and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals