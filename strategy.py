# Your Name: Test
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using daily KAMA for trend direction, RSI for momentum, and 
# Choppiness Index for regime filtering. Uses volume confirmation to filter weak signals.
# KAMA adapts to market efficiency, providing smooth trend signals that work in trending
# and ranging markets. RSI identifies overbought/oversold conditions, while Choppiness
# Index determines market regime: >61.8 = ranging (mean revert), <38.2 = trending (follow).
# Volume confirmation ensures only strong momentum signals are traded.
# Target: 25-50 trades/year with 0.30 position sizing to minimize fee drag.
# Works in bull/bear markets: KAMA captures trend, RSI catches reversals within trend,
# Choppiness filter avoids choppy markets where breakouts fail.

name = "4h_KAMA_RSI_Choppiness_Volume_v1"
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
    
    # Calculate daily KAMA for trend direction ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # KAMA calculation (Kaufman Adaptive Moving Average)
    close_1d = df_1d['close'].values
    direction = np.abs(close_1d[10:] - close_1d[:-10])
    volatility = np.sum(np.abs(np.diff(close_1d, axis=0)), axis=0)
    er = np.where(volatility != 0, direction / volatility, 0)
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    kama = np.full_like(close_1d, np.nan)
    kama[9] = close_1d[9]  # Seed
    for i in range(10, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    kama_trend = kama > np.roll(kama, 1)  # Rising KAMA = uptrend
    
    # Align daily KAMA trend to 4h timeframe
    kama_trend_aligned = align_htf_to_ltf(prices, df_1d, kama_trend.astype(float))
    
    # Calculate daily RSI for momentum
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([np.full(14, np.nan), rsi])
    
    # Align daily RSI to 4h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Calculate daily Choppiness Index for regime filtering
    atr_1d = pd.Series(np.maximum(
        high[1:] - low[:-1],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
    )).rolling(window=14).sum().values
    atr_1d = np.concatenate([np.full(13, np.nan), atr_1d])
    
    max_high = pd.Series(high).rolling(window=14).max().values
    min_low = pd.Series(low).rolling(window=14).min().values
    range_max_min = max_high - min_low
    
    chop = np.where(
        (range_max_min != 0) & (atr_1d != 0),
        100 * np.log10(atr_1d.sum() / range_max_min) / np.log10(14),
        50
    )
    chop = np.concatenate([np.full(13, np.nan), chop])
    
    # Align daily Choppiness to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume confirmation: >1.5x 20-period average on 4h
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(kama_trend_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: KAMA uptrend + RSI oversold + chop < 38.2 (trending) + volume
            if (kama_trend_aligned[i] and 
                rsi_aligned[i] < 30 and 
                chop_aligned[i] < 38.2 and 
                volume_filter[i]):
                signals[i] = 0.30
                position = 1
            # Short entry: KAMA downtrend + RSI overbought + chop < 38.2 (trending) + volume
            elif (not kama_trend_aligned[i] and 
                  rsi_aligned[i] > 70 and 
                  chop_aligned[i] < 38.2 and 
                  volume_filter[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: RSI overbought or KAMA trend change
            if rsi_aligned[i] > 70 or not kama_trend_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: RSI oversold or KAMA trend change
            if rsi_aligned[i] < 30 or kama_trend_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals