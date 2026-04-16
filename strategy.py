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
    
    # === 12h data (primary timeframe) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    volume_12h = df_12h['volume'].values
    
    # 12h Donchian(10) for entry/exit levels (tighter for fewer trades)
    high_10_12h = pd.Series(high_12h).rolling(window=10, min_periods=10).max().values
    low_10_12h = pd.Series(low_12h).rolling(window=10, min_periods=10).min().values
    donchian_upper_12h = align_htf_to_ltf(prices, df_12h, high_10_12h)
    donchian_lower_12h = align_htf_to_ltf(prices, df_12h, low_10_12h)
    
    # 12h ATR for volatility filter and stop
    tr1 = np.abs(high_12h - low_12h)
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr2[0] = np.inf
    tr3[0] = np.inf
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # === 1d data (HTF for trend filter) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # 1d EMA50 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d RSI for overbought/oversold filter
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_12h[i]) or np.isnan(donchian_lower_12h[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr_12h_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        price = close[i]
        upper_12h = donchian_upper_12h[i]
        lower_12h = donchian_lower_12h[i]
        ema_50_1d_val = ema_50_1d_aligned[i]
        atr_12h_val = atr_12h_aligned[i]
        rsi_1d_val = rsi_1d_aligned[i]
        
        # === STOPLOSS (close-based) ===
        if position == 1 and price < entry_price - 2.0 * atr_12h_val:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        elif position == -1 and price > entry_price + 2.0 * atr_12h_val:
            signals[i] = 0.0
            position = 0
            entry_price = 0.0
            continue
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below Donchian lower OR RSI becomes overbought
            if (price < lower_12h) or (rsi_1d_val > 70):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above Donchian upper OR RSI becomes oversold
            if (price > upper_12h) or (rsi_1d_val < 30):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above Donchian upper AND above daily EMA50 (trend filter) 
            # AND RSI not overbought
            if (price > upper_12h) and (price > ema_50_1d_val) and (rsi_1d_val < 60):
                signals[i] = 0.25
                position = 1
                entry_price = price
                continue
            
            # SHORT: Price breaks below Donchian lower AND below daily EMA50 (trend filter) 
            # AND RSI not oversold
            elif (price < lower_12h) and (price < ema_50_1d_val) and (rsi_1d_val > 40):
                signals[i] = -0.25
                position = -1
                entry_price = price
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_Donchian_Breakout_EMA50_1d_RSI"
timeframe = "12h"
leverage = 1.0