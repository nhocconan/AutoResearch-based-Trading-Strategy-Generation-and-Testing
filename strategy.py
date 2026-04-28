#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for multiple timeframe analysis
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Daily 20-period EMA for trend filter
    close_1d_series = pd.Series(df_1d['close'].values)
    ema20_1d = close_1d_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Daily 14-period RSI for momentum filter
    delta = close_1d_series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_14 = 100 - (100 / (1 + rs))
    rsi_14_values = rsi_14.fillna(50).values
    
    # Daily Bollinger Bands (20, 2) for volatility regime
    sma_20 = close_1d_series.rolling(window=20, min_periods=20).mean().values
    std_20 = close_1d_series.rolling(window=20, min_periods=20).std().values
    bb_upper = sma_20 + (2 * std_20)
    bb_lower = sma_20 - (2 * std_20)
    
    # Align all daily indicators to 4h timeframe
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    rsi_14_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_values)
    bb_upper_aligned = align_htf_to_ltf(prices, df_1d, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1d, bb_lower)
    
    # Volume filter: above average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Hour filter: 8-20 UTC (most active trading hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema20_1d_aligned[i]) or np.isnan(rsi_14_aligned[i]) or 
            np.isnan(bb_upper_aligned[i]) or np.isnan(bb_lower_aligned[i]) or 
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
        
        # Trend filter: price above/below daily EMA20
        trend_up = close[i] > ema20_1d_aligned[i]
        trend_down = close[i] < ema20_1d_aligned[i]
        
        # Momentum filter: RSI in favorable range
        rsi_favorable = (rsi_14_aligned[i] > 40) and (rsi_14_aligned[i] < 80)
        
        # Volatility filter: price within Bollinger Bands (not extreme)
        bb_filter = (close[i] >= bb_lower_aligned[i]) and (close[i] <= bb_upper_aligned[i])
        
        # Entry conditions: 
        # Long: pullback to EMA in uptrend with favorable momentum
        # Short: pullback to EMA in downtrend with favorable momentum
        long_pullback = (close[i] <= ema20_1d_aligned[i] * 1.005) and (close[i] >= ema20_1d_aligned[i] * 0.995)
        short_pullback = (close[i] <= ema20_1d_aligned[i] * 1.005) and (close[i] >= ema20_1d_aligned[i] * 0.995)
        
        long_entry = long_pullback and vol_filter and trend_up and rsi_favorable and bb_filter
        short_entry = short_pullback and vol_filter and trend_down and rsi_favorable and bb_filter
        
        # Exit conditions: opposite EMA touch or RSI extreme
        long_exit = (close[i] < ema20_1d_aligned[i] * 0.98) or (rsi_14_aligned[i] > 75)
        short_exit = (close[i] > ema20_1d_aligned[i] * 1.02) or (rsi_14_aligned[i] < 25)
        
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

name = "4h_EMAPullback_RSI_BB_Filter"
timeframe = "4h"
leverage = 1.0