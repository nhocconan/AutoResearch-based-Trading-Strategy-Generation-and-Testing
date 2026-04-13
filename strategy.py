# 40150
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Get 1d data for daily indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 200-day EMA for trend filter
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate daily ATR for volatility filter
    tr_1d = np.maximum(
        high_1d - low_1d,
        np.maximum(
            np.abs(high_1d - np.roll(close_1d, 1)),
            np.abs(low_1d - np.roll(close_1d, 1))
        )
    )
    tr_1d[0] = high_1d[0] - low_1d[0]
    atr_1d = np.zeros_like(tr_1d)
    for i in range(len(tr_1d)):
        if i < 14:
            atr_1d[i] = np.mean(tr_1d[:i+1])
        else:
            atr_1d[i] = 0.93 * atr_1d[i-1] + 0.07 * tr_1d[i]
    
    # Get 1w data for weekly trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to daily timeframe
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    # Track entry price for stoploss
    entry_price = np.full(n, np.nan)
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: daily close above/below EMA200
        uptrend = close[i] > ema_200_1d_aligned[i]
        downtrend = close[i] < ema_200_1d_aligned[i]
        
        # Weekly trend filter: weekly EMA50 slope
        weekly_uptrend = ema_50_1w_aligned[i] > ema_50_1w_aligned[i-5] if i >= 5 else False
        weekly_downtrend = ema_50_1w_aligned[i] < ema_50_1w_aligned[i-5] if i >= 5 else False
        
        # Volatility filter: avoid extremely low volatility days
        low_vol_filter = atr_1d_aligned[i] > 0.01 * close[i]  # ATR > 1% of price
        
        # Entry conditions: aligned weekly and daily trend with volatility filter
        long_entry = uptrend and weekly_uptrend and low_vol_filter
        short_entry = downtrend and weekly_downtrend and low_vol_filter
        
        # Exit conditions: trend reversal
        long_exit = not uptrend or not weekly_uptrend
        short_exit = not downtrend or not weekly_downtrend
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
            entry_price[i] = close[i]
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
            entry_price[i] = close[i]
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
            entry_price[i] = np.nan
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
            entry_price[i] = np.nan
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
                entry_price[i] = entry_price[i-1] if i > 0 else np.nan
            elif position == -1:
                signals[i] = -position_size
                entry_price[i] = entry_price[i-1] if i > 0 else np.nan
            else:
                signals[i] = 0.0
                entry_price[i] = np.nan
    
    return signals

name = "1d_1w_ema_trend_filter_v1"
timeframe = "1d"
leverage = 1.0