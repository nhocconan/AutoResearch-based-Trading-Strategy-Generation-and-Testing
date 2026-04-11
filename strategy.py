#!/usr/bin/env python3
"""
1h_4h_1d_camarilla_breakout_volume_v1
Strategy: 1h Camarilla pivot breakout with volume confirmation and 4h/1d trend filter
Timeframe: 1h
Leverage: 1.0
Hypothesis: Uses 1h price breakout above/below Camarilla pivot levels (H3/L3) confirmed by volume spike (>1.5x average volume) and filtered by 4h/1d EMA50 trend direction. Designed to capture strong momentum moves in trending markets while avoiding false breakouts in chop. Works in bull markets (breakouts with trend) and bear markets (breakouts against trend filtered out). Target: 60-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_camarilla_breakout_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        
        # Calculate Camarilla levels for current 1h bar
        # Using previous day's OHLC (approximated with current bar's previous values)
        # In practice, Camarilla uses prior day's OHLC, but we'll use rolling window
        if i >= 24:  # Need at least 24 hours of data for prior day approximation
            # Approximate prior day's OHLC using 24-period lookback
            prior_high = np.max(high[i-24:i])
            prior_low = np.min(low[i-24:i])
            prior_close = close[i-1]
            
            # Camarilla levels
            range_val = prior_high - prior_low
            h3 = prior_close + (range_val * 1.1 / 4)
            l3 = prior_close - (range_val * 1.1 / 4)
            h4 = prior_close + (range_val * 1.1 / 2)
            l4 = prior_close - (range_val * 1.1 / 2)
        else:
            # Not enough data yet
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Trend filters: price above/below 4h and 1d EMA50
        uptrend_4h = price_close > ema_50_4h_aligned[i]
        downtrend_4h = price_close < ema_50_4h_aligned[i]
        uptrend_1d = price_close > ema_50_1d_aligned[i]
        downtrend_1d = price_close < ema_50_1d_aligned[i]
        
        # Require both timeframes to agree on trend
        uptrend = uptrend_4h and uptrend_1d
        downtrend = downtrend_4h and downtrend_1d
        
        # Breakout conditions
        breakout_up = price_high > h3  # Break above H3
        breakout_down = price_low < l3  # Break below L3
        
        # Volume confirmation
        vol_confirmed = vol_spike[i]
        
        # Long: upward breakout with volume in uptrend
        long_signal = breakout_up and vol_confirmed and uptrend
        
        # Short: downward breakout with volume in downtrend
        short_signal = breakout_down and vol_confirmed and downtrend
        
        # Exit when price returns to midpoint (prior close)
        exit_long = position == 1 and price_close < prior_close
        exit_short = position == -1 and price_close > prior_close
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals