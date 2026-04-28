#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for 120 EMA trend filter and volume MA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 120:
        return np.zeros(n)
    
    # Daily EMA120 for trend filter
    close_1d_series = pd.Series(df_1d['close'].values)
    ema120_1d = close_1d_series.ewm(span=120, adjust=False, min_periods=120).mean().values
    ema120_1d_aligned = align_htf_to_ltf(prices, df_1d, ema120_1d)
    
    # Daily volume MA(20) for volume filter
    volume_1d_series = pd.Series(df_1d['volume'].values)
    vol_ma_1d = volume_1d_series.rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Get 6-hour RSI(14) for mean reversion entries
    close_series = pd.Series(close)
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.fillna(100).values
    
    # Session filter: 8-20 UTC (most active trading hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 120  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema120_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(rsi_values[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 8-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            # Outside session: flatten position
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter: current volume above daily average
        vol_filter = volume[i] > vol_ma_1d_aligned[i]
        
        # Trend filter: price above/below daily EMA120
        trend_up = close[i] > ema120_1d_aligned[i]
        trend_down = close[i] < ema120_1d_aligned[i]
        
        # RSI conditions for mean reversion
        rsi_oversold = rsi_values[i] < 30
        rsi_overbought = rsi_values[i] > 70
        
        # Entry conditions:
        # Long: RSI oversold + above daily EMA120 + volume
        # Short: RSI overbought + below daily EMA120 + volume
        long_entry = rsi_oversold and trend_up and vol_filter
        short_entry = rsi_overbought and trend_down and vol_filter
        
        # Exit conditions: RSI returns to neutral zone (40-60)
        long_exit = rsi_values[i] > 40 and position == 1
        short_exit = rsi_values[i] < 60 and position == -1
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_RSI14_EMA120_Volume_MeanReversion"
timeframe = "6h"
leverage = 1.0