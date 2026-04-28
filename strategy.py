#2025-06-14: 4h RSI Mean Reversion with Volume Confirmation and Trend Filter
#Hypothesis: RSI mean reversion works in both bull and bear markets when filtered by trend and volume.
#In bull markets, buy oversold dips in uptrends; in bear markets, sell overbought rallies in downtrends.
#Volume confirms conviction; RSI avoids chasing extremes.
#Timeframe: 4h balances trade frequency (~20-50/year) and signal quality.
#Uses 1d trend filter (EMA50) for multi-timeframe alignment.

#!/usr/bin/env python3
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily EMA50 for trend filter
    close_1d_series = pd.Series(df_1d['close'].values)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # RSI(14) on 4h close
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: above average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Hour filter: 8-20 UTC (most active trading hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ma[i])):
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
        
        # Volume filter: above average volume
        vol_filter = volume[i] > vol_ma[i]
        
        # Trend filter: price above/below daily EMA50
        trend_up = close[i] > ema50_1d_aligned[i]
        trend_down = close[i] < ema50_1d_aligned[i]
        
        # Entry conditions: 
        # Long: RSI < 30 (oversold) in uptrend + volume
        # Short: RSI > 70 (overbought) in downtrend + volume
        long_entry = (rsi[i] < 30) and vol_filter and trend_up
        short_entry = (rsi[i] > 70) and vol_filter and trend_down
        
        # Exit conditions: RSI returns to neutral zone (40-60)
        long_exit = (rsi[i] >= 40) and position == 1
        short_exit = (rsi[i] <= 60) and position == -1
        
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

name = "4h_RSI_MeanReversion_DailyTrend_Volume"
timeframe = "4h"
leverage = 1.0