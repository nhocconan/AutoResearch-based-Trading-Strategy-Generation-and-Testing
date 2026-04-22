#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h KAMA trend with RSI momentum and volume filter
# Uses Kaufman Adaptive Moving Average (KAMA) to capture trend while avoiding whipsaws
# RSI(14) confirms momentum direction, volume spike validates strength
# Trend-following only - avoids counter-trend losses in ranging markets
# Target: 20-30 trades/year per symbol (80-120 total) with strict entry conditions

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data for KAMA calculation (primary timeframe)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Load 1-day data for trend confirmation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate KAMA (2-period ER, 30-period smoothing constant)
    # ER = Efficiency Ratio = |net change| / sum(|changes|)
    change = np.abs(np.diff(close_4h, prepend=close_4h[0]))
    volatility = np.sum(np.lib.stride_tricks.sliding_window_view(change, 10), axis=1)
    # Pad volatility to match length
    volatility = np.concatenate([np.full(9, np.nan), volatility])
    net_change = np.abs(np.subtract(close_4h, np.roll(close_4h, 10)))
    er = np.divide(net_change, volatility, out=np.full_like(net_change, np.nan), where=volatility!=0)
    # Smoothing constants: fastest = 2/(2+1)=0.667, slowest = 2/(30+1)=0.0645
    sc = (er * 0.603 + 0.0645) ** 2
    # Handle NaN values
    sc = np.nan_to_num(sc, nan=0.0645**2)
    # Calculate KAMA
    kama = np.full_like(close_4h, np.nan)
    kama[9] = close_4h[9]  # Start after lookback period
    for i in range(10, len(close_4h)):
        if not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close_4h[i] - kama[i-1])
    
    # Calculate RSI(14) on 4h
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 50-period EMA on 1d for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume spike filter (20-period on 4h)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma20
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Align indicators to 4-hour timeframe
    kama_aligned = align_htf_to_ltf(prices, df_4h, kama)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):
        # Skip if data not ready or outside session
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above KAMA + RSI > 50 + volume spike + uptrend (price > 1d EMA50)
            if (close[i] > kama_aligned[i] and rsi[i] > 50 and vol_spike[i] and close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price below KAMA + RSI < 50 + volume spike + downtrend (price < 1d EMA50)
            elif (close[i] < kama_aligned[i] and rsi[i] < 50 and vol_spike[i] and close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses KAMA in opposite direction
            if position == 1:
                if close[i] < kama_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > kama_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_KAMA_RSI_Volume_Trend"
timeframe = "4h"
leverage = 1.0