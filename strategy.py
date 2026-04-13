#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1h RSI(2) mean reversion + 4h trend filter + session filter
    # Long: RSI(2) < 10 AND price > 4h EMA50 (uptrend) AND 08-20 UTC
    # Short: RSI(2) > 90 AND price < 4h EMA50 (downtrend) AND 08-20 UTC
    # Exit: RSI(2) crosses 50
    # Uses 4h EMA50 for trend filter (direction), 1h RSI(2) for timing
    # Session filter reduces noise outside active hours
    # Discrete position sizing (0.20) to minimize fee churn
    # Target: 60-150 total trades over 4 years (~15-37/year) to stay within limits
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    open_time = prices['open_time'].values
    
    # Get 4h data for EMA50 (call ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h EMA50 to 1h (wait for completed 4h bar)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate 1h RSI(2)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    def wilders_smoothing(data, period):
        alpha = 1.0 / period
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nanmean(data[:period])
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    avg_gain = wilders_smoothing(gain, 2)
    avg_loss = wilders_smoothing(loss, 2)
    
    rs = np.full_like(avg_gain, np.nan)
    mask = ~np.isnan(avg_loss) & (avg_loss > 0)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    
    rsi = np.full_like(avg_gain, 100.0)
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    rsi[np.isnan(avg_loss) & (avg_loss == 0)] = 100.0  # all gains
    rsi[np.isnan(avg_gain) & (avg_gain == 0)] = 0.0   # all losses
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(rsi[i]) or 
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        # RSI(2) signals
        rsi_oversold = rsi[i] < 10
        rsi_overbought = rsi[i] > 90
        rsi_exit = (rsi[i] >= 50 and position == 1) or (rsi[i] <= 50 and position == -1)
        
        # Trend filter: price vs 4h EMA50
        price_above_ema = close[i] > ema50_4h_aligned[i]
        price_below_ema = close[i] < ema50_4h_aligned[i]
        
        # Entry logic: RSI extreme + trend alignment + session
        long_entry = rsi_oversold and price_above_ema
        short_entry = rsi_overbought and price_below_ema
        
        # Exit logic: RSI crosses 50
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and rsi_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and rsi_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_4h_rsi2_mean_reversion_trend_filter_session_v1"
timeframe = "1h"
leverage = 1.0