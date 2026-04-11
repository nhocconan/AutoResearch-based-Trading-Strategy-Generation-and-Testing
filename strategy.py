#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_rsi_pullback_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d close for RSI
    close_1d = df_1d['close'].values
    
    # 14-period RSI on 1d close
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain = np.concatenate([[0], gain])
    loss = np.concatenate([[0], loss])
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Shift by 1 to use only completed 1d bars
    rsi_1d = np.roll(rsi_1d, 1)
    rsi_1d[0] = np.nan
    
    # Align 1d RSI to 4h timeframe
    rsi_1d_4h = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # 4h 20-period EMA for trend
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # 4h volume filter: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(rsi_1d_4h[i]) or np.isnan(ema_20[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_low = low[i]
        price_high = high[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        rsi_val = rsi_1d_4h[i]
        ema_val = ema_20[i]
        
        # Volume confirmation: higher threshold to reduce trades
        volume_confirmed = volume_current > 1.8 * vol_ma
        
        # Long conditions: RSI < 30 (oversold) and price near/above EMA20 with volume
        long_signal = volume_confirmed and (rsi_val < 30) and (price_low <= ema_val * 1.02)
        
        # Short conditions: RSI > 70 (overbought) and price near/below EMA20 with volume
        short_signal = volume_confirmed and (rsi_val > 70) and (price_high >= ema_val * 0.98)
        
        # Exit when RSI returns to neutral zone (40-60)
        exit_long = position == 1 and rsi_val >= 40
        exit_short = position == -1 and rsi_val <= 60
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: 1d RSI extremes on 4h timeframe with volume confirmation and EMA filter.
# Enters long when 1d RSI < 30 (oversold) with volume > 1.8x 20-period average and price near 4h EMA20.
# Enters short when 1d RSI > 70 (overbought) with same volume condition and price near 4h EMA20.
# Exits when RSI returns to neutral zone (40 for longs, 60 for shorts).
# Works in bull markets (buying oversold dips in uptrend) and bear markets (selling overbought rallies in downtrend).
# Uses strict RSI thresholds and high volume requirement to limit trades to ~20-30/year.