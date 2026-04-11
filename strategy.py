#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4d_ema_trend_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = prices.index.hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return signals
    
    # Calculate 4h EMA20 for trend filter
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Calculate 1h volume filter: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(20, n):
        # Skip if outside trading session
        if not session_mask[i]:
            signals[i] = 0.0
            continue
            
        # Skip if required data is invalid
        if np.isnan(ema_20_4h_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.8 * vol_ma
        
        # Trend alignment: price relative to 4h EMA20
        above_trend = price_close > ema_20_4h_aligned[i]
        below_trend = price_close < ema_20_4h_aligned[i]
        
        # Entry conditions
        long_signal = volume_confirmed and above_trend
        short_signal = volume_confirmed and below_trend
        
        # Exit when price crosses back through 4h EMA20
        exit_long = position == 1 and price_close < ema_20_4h_aligned[i]
        exit_short = position == -1 and price_close > ema_20_4h_aligned[i]
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals

# Hypothesis: 1h EMA trend following with 4h trend filter and volume confirmation.
# Uses 4h EMA20 as higher timeframe trend filter to ensure alignment with dominant trend.
# Enters long when 1h price > 4h EMA20 with volume confirmation (>1.8x average volume).
# Enters short when 1h price < 4h EMA20 with volume confirmation.
# Exits when price crosses back through 4h EMA20.
# Session filter (08-20 UTC) reduces noise during low-volume hours.
# Position size fixed at 0.20 to limit drawdown and reduce churn.
# Target: 60-150 total trades over 4 years (15-37/year) to minimize fee drag on 1h.
# The 4h trend filter prevents counter-trend trades, improving win rate in both bull and bear markets.
# Volume confirmation ensures institutional participation, reducing false breakouts.