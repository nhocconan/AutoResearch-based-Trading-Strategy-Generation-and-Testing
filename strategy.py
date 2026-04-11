#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy with 4h trend filter and daily volume confirmation.
# Uses 4h RSI for trend direction and 1d volume spike for institutional participation.
# Enters long when 4h RSI > 55 and 1h price closes above 1h EMA(20) with volume > 2x 1d average volume.
# Enters short when 4h RSI < 45 and 1h price closes below 1h EMA(20) with volume > 2x 1d average volume.
# Uses fixed position size of 0.20 to control risk. Designed for 15-30 trades/year on 1h timeframe.
# Volume filter ensures institutional participation, reducing false breakouts.
# 4h RSI filter prevents counter-trend trading in choppy markets.

name = "1h_4h_1d_rsi_volume_filter_v1"
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
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h RSI(14) for trend filter
    close_4h = df_4h['close'].values
    delta = np.diff(close_4h)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_14_4h = 100 - (100 / (1 + rs))
    # Prepend first value to match length
    rsi_14_4h = np.concatenate([[50.0], rsi_14_4h])
    
    # Align 4h RSI to 1h timeframe
    rsi_14_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_14_4h)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d volume moving average (20-period)
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d volume MA to 1h timeframe
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate 1h EMA(20) for entry timing
    ema_20_1h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after EMA period
        # Skip if any required data is invalid
        if (np.isnan(rsi_14_4h_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or
            np.isnan(ema_20_1h[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Volume filter: current volume > 2.0 * 1d average volume
        vol_filter = volume[i] > 2.0 * vol_ma_20_1d_aligned[i]
        
        # Determine 4h RSI trend direction
        is_bullish_trend = rsi_14_4h_aligned[i] > 55
        is_bearish_trend = rsi_14_4h_aligned[i] < 45
        
        # Entry conditions
        bullish_entry = (close[i] > ema_20_1h[i]) and vol_filter and is_bullish_trend
        bearish_entry = (close[i] < ema_20_1h[i]) and vol_filter and is_bearish_trend
        
        # Exit conditions: opposite entry signal
        exit_long = bearish_entry
        exit_short = bullish_entry
        
        # Priority: entry > exit > hold
        if bullish_entry and position != 1:
            position = 1
            signals[i] = 0.20
        elif bearish_entry and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals