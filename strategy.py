#/usr/bin/env python3
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
    
    # Get 1d data for weekly context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get weekly data for trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1d daily range (high-low) for volatility
    daily_range = high - low
    daily_range_ma_20 = pd.Series(daily_range).rolling(window=20, min_periods=20).mean().values
    
    # Calculate weekly EMA(20) for trend
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate weekly ATR(14) for volatility filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w_shift = np.roll(close_1w, 1)
    close_1w_shift[0] = close_1w[0]
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - close_1w_shift)
    tr3 = np.abs(low_1w - close_1w_shift)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1w = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_14_1w)
    
    # Calculate daily volume MA(20)
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need at least 20 days for daily range and weekly indicators
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(daily_range_ma_20[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(atr_14_1w_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        daily_range_now = daily_range[i]
        daily_range_ma = daily_range_ma_20[i]
        ema_trend = ema_20_1w_aligned[i]
        atr_vol = atr_14_1w_aligned[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        
        # Volatility filter: daily range > 1.5x weekly ATR
        vol_filter = daily_range_now > 1.5 * atr_vol
        
        # Volume filter: volume > 1.3x 20-day MA
        vol_ma_filter = vol_now > 1.3 * vol_ma
        
        # Trend filter: price above/below weekly EMA
        price_above_ema = close[i] > ema_trend
        price_below_ema = close[i] < ema_trend
        
        # Entry conditions
        if position == 0:
            # Long: price above weekly EMA + volatility expansion + volume surge
            if price_above_ema and vol_filter and vol_ma_filter:
                signals[i] = size
                position = 1
            # Short: price below weekly EMA + volatility expansion + volume surge
            elif price_below_ema and vol_filter and vol_ma_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below weekly EMA or volatility contraction
            if close[i] < ema_trend or daily_range_now < 0.8 * atr_vol:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above weekly EMA or volatility contraction
            if close[i] > ema_trend or daily_range_now < 0.8 * atr_vol:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_WeeklyEMA_VolatilityBreakout"
timeframe = "1d"
leverage = 1.0