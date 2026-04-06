#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA(20) trend + volume confirmation + ATR stoploss
# Long when price breaks above 20-day high AND price > 1w EMA(20) AND volume > 2x average
# Short when price breaks below 20-day low AND price < 1w EMA(20) AND volume > 2x average
# Exit when price returns to 10-day moving average or ATR-based stoploss hit
# Uses daily timeframe to minimize trades (target: 75-200 total over 4 years)
# Works in bull/bear markets by requiring trend alignment (1w EMA) and volume confirmation

name = "1d_donchian20_1w_ema_vol_v1"
timeframe = "1d"
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
    
    # Donchian Channel (20-period) on daily timeframe
    df_1d = get_htf_data(prices, '1d')
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate Donchian levels
    highest_high = pd.Series(daily_high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(daily_low).rolling(window=20, min_periods=20).min()
    
    # Align to 1d timeframe (already aligned since we're using 1d data)
    donchian_high = highest_high.values
    donchian_low = lowest_low.values
    
    # 1-week EMA(20) for trend filter
    df_1w = get_htf_data(prices, '1w')
    weekly_close = df_1w['close'].values
    ema_20 = pd.Series(weekly_close).ewm(span=20, min_periods=20, adjust=False).mean()
    ema_20_aligned = align_htf_to_ltf(prices, df_1w, ema_20.values)
    
    # Volume confirmation: volume > 2x 20-day average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 2.0 * volume_ma.values
    
    # ATR for stoploss (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(ema_20_aligned[i]) or np.isnan(volume_threshold[i]) or np.isnan(atr[i]):
            if position != 0:
                # Hold position until exit conditions
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price returns to 10-day moving average OR stoploss hit
            ma_10 = pd.Series(daily_close).rolling(window=10, min_periods=10).mean()
            ma_10_aligned = align_htf_to_ltf(prices, df_1d, ma_10.values)
            if daily_close[i] <= ma_10_aligned[i] or close[i] < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price returns to 10-day moving average OR stoploss hit
            ma_10 = pd.Series(daily_close).rolling(window=10, min_periods=10).mean()
            ma_10_aligned = align_htf_to_ltf(prices, df_1d, ma_10.values)
            if daily_close[i] >= ma_10_aligned[i] or close[i] > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + trend alignment + volume confirmation
            # Long: price breaks above 20-day high AND price > 1w EMA(20) AND volume confirmation
            if daily_close[i] > donchian_high[i] and daily_close[i] > ema_20_aligned[i] and volume[i] > volume_threshold[i]:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below 20-day low AND price < 1w EMA(20) AND volume confirmation
            elif daily_close[i] < donchian_low[i] and daily_close[i] < ema_20_aligned[i] and volume[i] > volume_threshold[i]:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals