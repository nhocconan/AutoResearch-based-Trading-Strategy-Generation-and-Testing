#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour RSI(7) with 4-hour Donchian(20) and 1-day trend filter
# Long when RSI(7) < 30 + price > 4h Donchian upper (breakout in oversold) + 1-day close > SMA50 (uptrend)
# Short when RSI(7) > 70 + price < 4h Donchian lower (breakdown in overbought) + 1-day close < SMA50 (downtrend)
# Exit when RSI crosses 50 (mean reversion) or price crosses opposite Donchian band
# Session filter: 08-20 UTC to avoid low-volume Asian session
# Position size: 0.20 (20% of capital)
# Target: 80-160 total trades over 4 years (20-40/year)

name = "1h_rsi7_4h_donchian_1d_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 4-hour data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4-hour Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    high_4h_s = pd.Series(high_4h)
    low_4h_s = pd.Series(low_4h)
    donchian_upper = high_4h_s.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_4h_s.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1h
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    
    # 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1-day SMA(50)
    close_1d = df_1d['close'].values
    close_1d_s = pd.Series(close_1d)
    sma50_1d = close_1d_s.rolling(window=50, min_periods=50).mean().values
    sma50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma50_1d)
    
    # 1-hour RSI(7)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_s = pd.Series(gain)
    loss_s = pd.Series(loss)
    avg_gain = gain_s.ewm(alpha=1/7, adjust=False, min_periods=7).mean().values
    avg_loss = loss_s.ewm(alpha=1/7, adjust=False, min_periods=7).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Session filter: 08-20 UTC (already datetime64[ms])
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(rsi[i]) or np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or np.isnan(sma50_1d_aligned[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if position == 1:  # long position
            # Exit: RSI crosses 50 or price breaks below Donchian lower
            if rsi[i] >= 50 or close[i] < donchian_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: RSI crosses 50 or price breaks above Donchian upper
            if rsi[i] <= 50 or close[i] > donchian_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.20
        else:
            # Look for entries only during session
            if in_session:
                # Long: RSI oversold + price breaks above Donchian upper + 1-day uptrend
                if rsi[i] < 30 and close[i] > donchian_upper_aligned[i] and close[i] > sma50_1d_aligned[i]:
                    signals[i] = 0.20
                    position = 1
                    entry_price = close[i]
                # Short: RSI overbought + price breaks below Donchian lower + 1-day downtrend
                elif rsi[i] > 70 and close[i] < donchian_lower_aligned[i] and close[i] < sma50_1d_aligned[i]:
                    signals[i] = -0.20
                    position = -1
                    entry_price = close[i]
    
    return signals