#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian(20) breakout direction + 1h RSI(14) pullback entry.
# Long when 4h Donchian upper is broken on 1h and RSI(14) < 40 (oversold pullback in uptrend).
# Short when 4h Donchian lower is broken on 1h and RSI(14) > 60 (overbought pullback in downtrend).
# Uses discrete position size 0.20. Session filter 08-20 UTC to avoid low-volume hours.
# Designed to work in both bull (breakouts with pullbacks) and bear (fades of rallies into resistance).
# Target: 15-37 trades/year/symbol to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 4h data once before loop for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # === 4h Indicators: Donchian Channels (20-period) ===
    donchian_upper_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_lower_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align 4h indicators to 1h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper_4h)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower_4h)
    
    # === 1h Indicators: RSI(14) ===
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral when undefined
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # open_time is datetime64[ms], index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 20  # Donchian and RSI need 20 periods
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            position = 0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or
            np.isnan(rsi[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        upper = donchian_upper_aligned[i]
        lower = donchian_lower_aligned[i]
        price = close[i]
        rsi_val = rsi[i]
        
        # === EXIT LOGIC: reverse Donchian break ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit on break below 4h Donchian lower (trend invalidation)
            if price < lower:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit on break above 4h Donchian upper (trend invalidation)
            if price > upper:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: price breaks above 4h Donchian upper AND RSI < 40 (pullback in uptrend)
            if price > upper and rsi_val < 40:
                signals[i] = 0.20
                position = 1
            
            # SHORT: price breaks below 4h Donchian lower AND RSI > 60 (pullback in downtrend)
            elif price < lower and rsi_val > 60:
                signals[i] = -0.20
                position = -1
        
        else:
            signals[i] = position * 0.20  # maintain position
    
    return signals

name = "1h_4hDonchian20_RSI14_Pullback_Session08-20_v1"
timeframe = "1h"
leverage = 1.0