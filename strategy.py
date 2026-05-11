#!/usr/bin/env python3
name = "1d_KAMA_Direction_RSI_ChopFilter_v2"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA direction (10-day ER) - corrected calculation
    diff = np.diff(close, prepend=close[0])
    price_diff = np.abs(diff)
    volatility = np.abs(diff)
    price_diff_series = pd.Series(price_diff)
    volatility_series = pd.Series(volatility)
    er = price_diff_series.rolling(window=10, min_periods=10).sum() / volatility_series.rolling(window=10, min_periods=10).sum()
    er = er.fillna(0).values
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # smooth constant
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    kama_dir = kama > np.roll(kama, 1)  # today's KAMA > yesterday's
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_series = pd.Series(gain)
    loss_series = pd.Series(loss)
    avg_gain = gain_series.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss_series.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Chopiness index (14-day)
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_series = pd.Series(tr)
    atr = atr_series.rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    denominator = highest_high - lowest_low + 1e-10
    chop = 100 * np.log10(atr / denominator) / np.log10(14)
    chop = np.nan_to_num(chop, nan=50.0)
    
    # Weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    sma_50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    weekly_uptrend = close > sma_50_1w_aligned
    
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.5 * volume_ma20
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(20, 14, 10)  # ensure all indicators ready
    
    for i in range(start_idx, n):
        if np.isnan(kama_dir[i]) or np.isnan(rsi[i]) or np.isnan(chop[i]) or np.isnan(weekly_uptrend[i]) or np.isnan(volume_ma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: KAMA up, RSI > 50, chop < 61.8 (trending), weekly uptrend, volume
            if kama_dir[i] and rsi[i] > 50 and chop[i] < 61.8 and weekly_uptrend[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: KAMA down, RSI < 50, chop < 61.8, weekly downtrend, volume
            elif not kama_dir[i] and rsi[i] < 50 and chop[i] < 61.8 and not weekly_uptrend[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: KAMA down or chop > 61.8 (range) or weekly downtrend
            if not kama_dir[i] or chop[i] > 61.8 or not weekly_uptrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: KAMA up or chop > 61.8 (range) or weekly uptrend
            if kama_dir[i] or chop[i] > 61.8 or weekly_uptrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals