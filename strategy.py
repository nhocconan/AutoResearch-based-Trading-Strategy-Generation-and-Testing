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
    
    # Get 1d data for trend and volatility regime
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d ATR(14) for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_shift = np.roll(close_1d, 1)
    close_1d_shift[0] = close_1d[0]
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - close_1d_shift)
    tr3 = np.abs(low_1d - close_1d_shift)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 4h Donchian(20) for entry signals
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    upper = np.zeros(len(high_4h))
    lower = np.zeros(len(low_4h))
    for i in range(20, len(high_4h)):
        upper[i] = np.max(high_4h[i-20:i])
        lower[i] = np.min(low_4h[i-20:i])
    upper[:20] = np.nan
    lower[:20] = np.nan
    donch_upper_4h = upper
    donch_lower_4h = lower
    donch_upper_4h_aligned = align_htf_to_ltf(prices, df_4h, donch_upper_4h)
    donch_lower_4h_aligned = align_htf_to_ltf(prices, df_4h, donch_lower_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need 4h Donchian and 1d indicators
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donch_upper_4h_aligned[i]) or np.isnan(donch_lower_4h_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr_14_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        upper = donch_upper_4h_aligned[i]
        lower = donch_lower_4h_aligned[i]
        ema_trend = ema_34_1d_aligned[i]
        atr_vol = atr_14_1d_aligned[i]
        
        # Volatility filter: only trade when volatility is elevated (ATR > 1.5 * 20-period MA of ATR)
        if i >= 20:
            atr_ma_20 = np.nanmean(atr_14_1d_aligned[i-20:i]) if not np.isnan(np.nanmean(atr_14_1d_aligned[i-20:i])) else atr_vol
            vol_filter = atr_vol > 1.5 * atr_ma_20
        else:
            vol_filter = True  # Not enough data for MA, allow trade
        
        # Entry conditions: breakout with trend alignment and volatility filter
        if position == 0:
            # Long: break above upper band + above 1d EMA + volatility filter
            if close[i] > upper and close[i] > ema_trend and vol_filter:
                signals[i] = size
                position = 1
            # Short: break below lower band + below 1d EMA + volatility filter
            elif close[i] < lower and close[i] < ema_trend and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: close below 1d EMA or ATR drops below threshold
            if close[i] < ema_trend or atr_vol < 0.5 * atr_ma_20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: close above 1d EMA or ATR drops below threshold
            if close[i] > ema_trend or atr_vol < 0.5 * atr_ma_20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_1dEMA34_ATRVolatilityFilter"
timeframe = "4h"
leverage = 1.0