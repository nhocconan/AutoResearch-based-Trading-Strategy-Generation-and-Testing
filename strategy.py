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
    
    # === 4h data (primary) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # 4-hour ATR for volatility
    high_4h_series = pd.Series(high_4h)
    low_4h_series = pd.Series(low_4h)
    close_4h_series = pd.Series(close_4h)
    tr1 = high_4h_series - low_4h_series
    tr2 = (high_4h_series - close_4h_series.shift(1)).abs()
    tr3 = (low_4h_series - close_4h_series.shift(1)).abs()
    tr2.iloc[0] = np.inf
    tr3.iloc[0] = np.inf
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_4h = tr.rolling(window=14, min_periods=14).mean().values
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    # 4-hour EMA20 for trend
    ema_20_4h = close_4h_series.ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # === 12h data (HTF for direction) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # 12-hour EMA50 for trend filter
    close_12h_series = pd.Series(close_12h)
    ema_50_12h = close_12h_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # === RSI on 4h for momentum ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # === Volume spike detection on 4h ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma_20
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_20_4h_aligned[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(atr_4h_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        price = close[i]
        ema_20_4h_val = ema_20_4h_aligned[i]
        ema_50_12h_val = ema_50_12h_aligned[i]
        atr_4h_val = atr_4h_aligned[i]
        rsi_val = rsi[i]
        vol_ratio_val = vol_ratio[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below EMA20 OR RSI becomes overbought
            if (price < ema_20_4h_val) or (rsi_val > 70):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above EMA20 OR RSI becomes oversold
            if (price > ema_20_4h_val) or (rsi_val < 30):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Only trade during session
            if in_session:
                # LONG: Price above EMA20 AND above EMA50 (12h) 
                # AND RSI not overbought AND volume spike
                if (price > ema_20_4h_val) and (price > ema_50_12h_val) and (rsi_val < 60) and \
                   (vol_ratio_val > 2.0):
                    signals[i] = 0.25
                    position = 1
                    continue
                
                # SHORT: Price below EMA20 AND below EMA50 (12h) 
                # AND RSI not oversold AND volume spike
                elif (price < ema_20_4h_val) and (price < ema_50_12h_val) and (rsi_val > 40) and \
                     (vol_ratio_val > 2.0):
                    signals[i] = -0.25
                    position = -1
                    continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "4h_EMA20_EMA50_12h_RSI_Volume"
timeframe = "4h"
leverage = 1.0