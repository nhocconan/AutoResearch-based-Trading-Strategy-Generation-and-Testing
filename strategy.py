#!/usr/bin/env python3
# 1h_ema_macd_vol_regime_v1
# Hypothesis: 1h strategy using 4h EMA for trend direction, MACD histogram for momentum,
# and volume spike for confirmation. Only trade during 08-20 UTC session.
# Enters long when 4h EMA up, MACD histogram positive and rising, volume > 1.5x 20MA.
# Enters short when 4h EMA down, MACD histogram negative and falling, volume > 1.5x 20MA.
# Exits on opposite MACD crossover or volume dry-up.
# Uses discrete position sizing (0.20) to minimize fee churn.
# Target: 15-37 trades/year (60-150 total over 4 years) on BTC/ETH/SOL.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_ema_macd_vol_regime_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    open_ = prices['open'].values
    volume = prices['volume'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Get 4h data for trend direction (HTF)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    
    # 4h EMA(21) for trend
    close_4h = pd.Series(df_4h['close'].values)
    ema_4h = close_4h.ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # MACD histogram on 1h
    close_s = pd.Series(close)
    ema_fast = close_s.ewm(span=12, min_periods=12, adjust=False).mean()
    ema_slow = close_s.ewm(span=26, min_periods=26, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=9, min_periods=9, adjust=False).mean()
    macd_hist = macd_line - macd_signal
    macd_hist_values = macd_hist.values
    
    # MACD histogram slope (rising/falling)
    macd_hist_slope = np.diff(macd_hist_values, prepend=macd_hist_values[0])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if not in session or any required data is NaN
        if not in_session[i] or \
           np.isnan(ema_4h_aligned[i]) or np.isnan(macd_hist_values[i]) or \
           np.isnan(macd_hist_slope[i]) or np.isnan(volume_ma[i]) or \
           np.isnan(close[i]) or np.isnan(volume[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: MACD histogram crosses below zero or volume drops below average
            if macd_hist_values[i] <= 0 or volume[i] < volume_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: MACD histogram crosses above zero or volume drops below average
            if macd_hist_values[i] >= 0 or volume[i] < volume_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Check for entry conditions
            bullish = (ema_4h_aligned[i] > ema_4h_aligned[i-1] and  # 4h EMA rising
                      macd_hist_values[i] > 0 and                  # MACD histogram positive
                      macd_hist_slope[i] > 0 and                   # MACD histogram rising
                      volume_confirmed)                            # Volume spike
            
            bearish = (ema_4h_aligned[i] < ema_4h_aligned[i-1] and  # 4h EMA falling
                      macd_hist_values[i] < 0 and                  # MACD histogram negative
                      macd_hist_slope[i] < 0 and                   # MACD histogram falling
                      volume_confirmed)                            # Volume spike
            
            if bullish:
                position = 1
                signals[i] = 0.20
            elif bearish:
                position = -1
                signals[i] = -0.20
    
    return signals