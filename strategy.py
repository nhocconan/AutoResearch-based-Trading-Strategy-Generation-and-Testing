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
    
    # Get daily data for indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily EMA50 for trend filter
    close_1d_series = pd.Series(df_1d['close'].values)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Daily RSI(14) for momentum
    delta = pd.Series(df_1d['close'].values).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi14_1d = (100 - (100 / (1 + rs))).values
    
    # Daily Bollinger Bands (20, 2) for volatility regime
    ma20 = pd.Series(df_1d['close'].values).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(df_1d['close'].values).rolling(window=20, min_periods=20).std().values
    upper_bb = ma20 + 2 * std20
    lower_bb = ma20 - 2 * std20
    
    # Align to daily timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    rsi14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi14_1d)
    upper_bb_aligned = align_htf_to_ltf(prices, df_1d, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_1d, lower_bb)
    ma20_aligned = align_htf_to_ltf(prices, df_1d, ma20)
    
    # Volume filter: above 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Hour filter: 8-20 UTC (most active trading hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(rsi14_1d_aligned[i]) or 
            np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i]) or
            np.isnan(ma20_aligned[i]) or np.isnan(vol_ma[i])):
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
        
        # Momentum filter: RSI not extreme
        rsi_not_overbought = rsi14_1d_aligned[i] < 70
        rsi_not_oversold = rsi14_1d_aligned[i] > 30
        
        # Volatility regime: price within Bollinger Bands (not too volatile)
        in_bb = (close[i] >= lower_bb_aligned[i]) and (close[i] <= upper_bb_aligned[i])
        
        # Entry conditions: 
        # Long: pullback to EMA50 in uptrend with good momentum
        # Short: pullback to EMA50 in downtrend with good momentum
        near_ema = abs(close[i] - ema50_1d_aligned[i]) / ema50_1d_aligned[i] < 0.02  # Within 2% of EMA
        
        long_entry = near_ema and trend_up and vol_filter and rsi_not_overbought and in_bb
        short_entry = near_ema and trend_down and vol_filter and rsi_not_oversold and in_bb
        
        # Exit conditions: opposite signal or RSI extreme
        long_exit = (rsi14_1d_aligned[i] >= 70) or (close[i] > upper_bb_aligned[i])
        short_exit = (rsi14_1d_aligned[i] <= 30) or (close[i] < lower_bb_aligned[i])
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
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

name = "1d_EMA50_Pullback_RSI_BB_Volume_Session"
timeframe = "1d"
leverage = 1.0