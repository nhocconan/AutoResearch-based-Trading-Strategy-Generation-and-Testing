#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Keltner Channel breakout with weekly trend filter and volume confirmation
# Works in bull markets (breakouts continue trends) and bear markets (mean reversion at extremes)
# Uses 1d primary timeframe with 1h trend filter for better timing
# Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Keltner Channel (20, 1.5)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get 1h data for trend filter
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 20:
        return np.zeros(n)
    
    # 1d Keltner Channel components
    close_1d_series = pd.Series(df_1d['close'].values)
    high_1d_series = pd.Series(df_1d['high'].values)
    low_1d_series = pd.Series(df_1d['low'].values)
    
    # EMA20 for middle line
    ema20 = close_1d_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # ATR(10) for channel width
    tr1 = np.maximum(high_1d_series.iloc[1:].values, low_1d_series.iloc[:-1].values)
    tr1 = np.maximum(tr1, np.abs(close_1d_series.iloc[1:].values - close_1d_series.iloc[:-1].values))
    tr1 = np.concatenate([[0], tr1])
    tr2 = np.maximum(high_1d_series.values, low_1d_series.values)
    tr2 = np.maximum(tr2, np.abs(close_1d_series.values - close_1d_series.iloc[[0]].values))
    tr = np.maximum(tr1, tr2)
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    upper_keltner = ema20 + 1.5 * atr
    lower_keltner = ema20 - 1.5 * atr
    
    # 1h EMA50 for trend filter
    close_1h_series = pd.Series(df_1h['close'].values)
    ema50_1h = close_1h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align indicators to 1d timeframe
    upper_keltner_aligned = align_htf_to_ltf(prices, df_1d, upper_keltner)
    lower_keltner_aligned = align_htf_to_ltf(prices, df_1d, lower_keltner)
    ema20_aligned = align_htf_to_ltf(prices, df_1d, ema20)
    ema50_1h_aligned = align_htf_to_ltf(prices, df_1h, ema50_1h)
    
    # Volume filter: above average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Hour filter: 8-20 UTC (most active trading hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_keltner_aligned[i]) or np.isnan(lower_keltner_aligned[i]) or 
            np.isnan(ema20_aligned[i]) or np.isnan(ema50_1h_aligned[i]) or 
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
        
        # Trend filter: price above/below 1h EMA50
        trend_up = close[i] > ema50_1h_aligned[i]
        trend_down = close[i] < ema50_1h_aligned[i]
        
        # Entry conditions: 
        # Long: price breaks above upper Keltner Channel in uptrend + volume
        # Short: price breaks below lower Keltner Channel in downtrend + volume
        long_entry = (close[i] > upper_keltner_aligned[i]) and vol_filter and trend_up
        short_entry = (close[i] < lower_keltner_aligned[i]) and vol_filter and trend_down
        
        # Exit conditions: price returns to middle Keltner Channel (EMA20)
        long_exit = (close[i] < ema20_aligned[i]) and position == 1
        short_exit = (close[i] > ema20_aligned[i]) and position == -1
        
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

name = "1d_KeltnerBreakout_1hTrend_Volume_Session"
timeframe = "1d"
leverage = 1.0