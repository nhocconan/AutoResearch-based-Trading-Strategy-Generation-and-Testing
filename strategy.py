#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h Donchian(20) for trend direction
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    upper_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    upper_4h_aligned = align_htf_to_ltf(prices, df_4h, upper_4h)
    lower_4h_aligned = align_htf_to_ltf(prices, df_4h, lower_4h)
    
    # Calculate 4h ATR(14) for volatility filter
    tr_4h = np.maximum(
        high_4h[1:] - low_4h[1:],
        np.maximum(
            np.abs(high_4h[1:] - high_4h[:-1]),
            np.abs(low_4h[1:] - low_4h[:-1])
        )
    )
    tr_4h = np.concatenate([[np.nan], tr_4h])
    atr_4h = pd.Series(tr_4h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    # Calculate 1h ADX(14) for trend strength filter
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - high[:-1]),
            np.abs(low[1:] - low[:-1])
        )
    )
    tr = np.concatenate([[np.nan], tr])
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    dm_plus_smooth = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    dm_minus_smooth = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
    dx = (np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)) * 100
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20  # 20% position size
    
    # Start after enough data for calculations
    start = 200
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_4h_aligned[i]) or 
            np.isnan(lower_4h_aligned[i]) or 
            np.isnan(atr_4h_aligned[i]) or 
            np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        # Skip outside session
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Trend filter: price above/below 4h Donchian bands
        trend_long = price > upper_4h_aligned[i]
        trend_short = price < lower_4h_aligned[i]
        
        # Volatility filter: require sufficient volatility
        vol_filter = atr_4h_aligned[i] > 0.01 * price  # at least 1% of price
        
        # Trend strength filter: ADX > 25
        adx_filter = adx[i] > 25
        
        if position == 0:
            # Long entry: price above upper Donchian + volatility + trend strength
            if trend_long and vol_filter and adx_filter:
                position = 1
                signals[i] = position_size
            # Short entry: price below lower Donchian + volatility + trend strength
            elif trend_short and vol_filter and adx_filter:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below lower Donchian OR ADX < 20
            if price < lower_4h_aligned[i] or adx[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above upper Donchian OR ADX < 20
            if price > upper_4h_aligned[i] or adx[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_4hDonchian20_ATR_ADX_Filter_v1"
timeframe = "1h"
leverage = 1.0