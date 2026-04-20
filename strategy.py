#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 34-period EMA for trend bias
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate daily range (high - low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    daily_range = high_1d - low_1d
    # 20-period average of daily range (volatility measure)
    avg_daily_range = pd.Series(daily_range).rolling(window=20, min_periods=20).mean().values
    avg_daily_range_aligned = align_htf_to_ltf(prices, df_1d, avg_daily_range)
    
    # Calculate 6-period RSI on 6h timeframe
    close_6h = prices['close'].values
    delta = np.diff(close_6h, prepend=close_6h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    def wilder_smooth(data, period):
        result = np.zeros_like(data)
        alpha = 1.0 / period
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    avg_gain = wilder_smooth(gain, 6)
    avg_loss = wilder_smooth(loss, 6)
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_6h = 100 - (100 / (1 + rs))
    
    # Session filter: 8-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Session filter: only trade 8-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get values
        close_val = prices['close'].iloc[i]
        ema_34_val = ema_34_aligned[i]
        avg_range_val = avg_daily_range_aligned[i]
        rsi_val = rsi_6h[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema_34_val) or np.isnan(avg_range_val) or np.isnan(rsi_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Dynamic range filter: only trade when daily volatility is elevated
        # This helps avoid choppy markets
        if i >= 20:  # Need enough history for average range
            range_ratio = avg_range_val / np.mean(avg_daily_range_aligned[max(0, i-20):i+1]) if i >= 20 else 1.0
            low_vol_filter = range_ratio < 0.5  # Avoid extremely low volatility periods
        else:
            low_vol_filter = False
        
        if position == 0:
            # Long: price above daily EMA34 (bullish bias), RSI oversold (<30), not in extremely low volatility
            if close_val > ema_34_val and rsi_val < 30 and not low_vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price below daily EMA34 (bearish bias), RSI overbought (>70), not in extremely low volatility
            elif close_val < ema_34_val and rsi_val > 70 and not low_vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below daily EMA34 or RSI overbought (>70)
            if close_val < ema_34_val or rsi_val > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above daily EMA34 or RSI oversold (<30)
            if close_val > ema_34_val or rsi_val < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# 6h_EMA34_RSI_MeanReversion_VolatilityFilter_v1
# Uses daily EMA34 for trend bias (long above, short below)
# Uses 6-period RSI for mean reversion entries (oversold/overbought)
# Includes volatility filter to avoid extremely low volatility periods
# Session filter: 8-20 UTC to avoid low-volume periods
# Designed for 6h timeframe with ~15-30 trades/year
name = "6h_EMA34_RSI_MeanReversion_VolatilityFilter_v1"
timeframe = "6h"
leverage = 1.0