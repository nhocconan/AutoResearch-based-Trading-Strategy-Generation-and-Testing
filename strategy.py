#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour strategy using 4-hour trend (EMA50) and 1-day regime (Choppiness Index) for direction,
# with 1-hour momentum (RSI) for entry timing. Long in bullish trending/low-chop markets when RSI crosses above 50;
# short in bearish trending/low-chop markets when RSI crosses below 50. Uses session filter (08-20 UTC) to avoid low-liquidity hours.
# Designed for low trade frequency (target: 15-37/year) to minimize fee drag and improve generalization in both bull and bear markets.
name = "1h_4hEMA50_1dChop_RSI50"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1-hour RSI for entry timing
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # 4-hour EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_rising = np.zeros_like(ema_50_4h, dtype=bool)
    ema_50_falling = np.zeros_like(ema_50_4h, dtype=bool)
    ema_50_rising[1:] = ema_50_4h[1:] > ema_50_4h[:-1]
    ema_50_falling[1:] = ema_50_4h[1:] < ema_50_4h[:-1]
    ema_50_rising_aligned = align_htf_to_ltf(prices, df_4h, ema_50_rising)
    ema_50_falling_aligned = align_htf_to_ltf(prices, df_4h, ema_50_falling)
    
    # 1-day Choppiness Index for regime filter (low chop = trending)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    atr_1d = np.zeros(len(close_1d))
    for i in range(1, len(close_1d)):
        tr = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d[i-1]), abs(low_1d[i] - close_1d[i-1]))
        if i == 1:
            atr_1d[i] = tr
        else:
            atr_1d[i] = (atr_1d[i-1] * 13 + tr) / 14
    sum_high_low = np.zeros(len(close_1d))
    for i in range(14, len(close_1d)):
        sum_high_low[i] = np.sum(high_1d[i-13:i+1] - low_1d[i-13:i+1])
    chop = np.full(len(close_1d), 50.0)
    for i in range(14, len(close_1d)):
        if sum_high_low[i] > 0 and atr_1d[i] > 0:
            chop[i] = 100 * np.log10(sum_high_low[i] / (atr_1d[i] * 14)) / np.log10(14)
    chop_below_38 = chop < 38.2  # trending regime
    chop_below_38_aligned = align_htf_to_ltf(prices, df_1d, chop_below_38, additional_delay_bars=0)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(rsi[i]) or np.isnan(ema_50_rising_aligned[i]) or np.isnan(ema_50_falling_aligned[i]) or 
            np.isnan(chop_below_38_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if in_session[i]:
            if position == 0:
                # Long: RSI crosses above 50 AND 4h EMA50 rising AND 1d chop < 38.2 (trending)
                if i > 0 and rsi[i-1] <= 50 and rsi[i] > 50 and ema_50_rising_aligned[i] and chop_below_38_aligned[i]:
                    signals[i] = 0.20
                    position = 1
                # Short: RSI crosses below 50 AND 4h EMA50 falling AND 1d chop < 38.2 (trending)
                elif i > 0 and rsi[i-1] >= 50 and rsi[i] < 50 and ema_50_falling_aligned[i] and chop_below_38_aligned[i]:
                    signals[i] = -0.20
                    position = -1
            elif position == 1:
                # Exit: RSI crosses below 50
                if rsi[i] < 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            elif position == -1:
                # Exit: RSI crosses above 50
                if rsi[i] > 50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
        else:
            # Outside session: flatten if in position
            if position != 0:
                signals[i] = 0.0
                position = 0
    
    return signals