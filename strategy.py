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
    
    # Get daily data for higher timeframe analysis
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Daily ATR for volatility filter (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.maximum(np.abs(low_1d[1:] - close_1d[:-1]), tr1)
    tr = np.concatenate([[np.inf], tr2])  # First TR is undefined
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Weekly EMA20 for trend filter
    close_1w_series = pd.Series(df_1w['close'].values)
    ema20_1w = close_1w_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Daily range for volatility regime (20-period)
    daily_range = high_1d - low_1d
    range_ma = pd.Series(daily_range).rolling(window=20, min_periods=20).mean().values
    range_std = pd.Series(daily_range).rolling(window=20, min_periods=20).std().values
    # Avoid division by zero
    range_std_safe = np.where(range_std == 0, 1e-10, range_std)
    z_score_range = (daily_range - range_ma) / range_std_safe
    
    # Align daily indicators to 12h timeframe
    atr14_aligned = align_htf_to_ltf(prices, df_1d, atr14)
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    z_score_range_aligned = align_htf_to_ltf(prices, df_1d, z_score_range)
    
    # 12-hour Bollinger Bands (20, 2) for mean reversion signals
    close_12h_series = pd.Series(close)
    bb_mid = close_12h_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_12h_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    
    # Hour filter: 8-20 UTC (most active trading hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(atr14_aligned[i]) or np.isnan(ema20_1w_aligned[i]) or 
            np.isnan(z_score_range_aligned[i]) or np.isnan(bb_mid[i]) or 
            np.isnan(bb_upper[i]) or np.isnan(bb_lower[i])):
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
        
        # Volatility filter: only trade in high volatility regimes
        vol_filter = z_score_range_aligned[i] > 0.5  # Above average volatility
        
        # Trend filter: price above/below weekly EMA20
        trend_up = close[i] > ema20_1w_aligned[i]
        trend_down = close[i] < ema20_1w_aligned[i]
        
        # Mean reversion signals from 12h Bollinger Bands
        bb_break_lower = close[i] < bb_lower[i]  # Price below lower band
        bb_break_upper = close[i] > bb_upper[i]   # Price above upper band
        
        # Entry conditions:
        # Long: price breaks below lower BB in uptrend during high volatility (mean reversion long)
        # Short: price breaks above upper BB in downtrend during high volatility (mean reversion short)
        long_entry = bb_break_lower and vol_filter and trend_up
        short_entry = bb_break_upper and vol_filter and trend_down
        
        # Exit conditions: price returns to middle Bollinger Band
        long_exit = (close[i] > bb_mid[i]) and position == 1
        short_exit = (close[i] < bb_mid[i]) and position == -1
        
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

name = "12h_BollingerMeanReversion_WeeklyTrend_VolatilityFilter"
timeframe = "12h"
leverage = 1.0