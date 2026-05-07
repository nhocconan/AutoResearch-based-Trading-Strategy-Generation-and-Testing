#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian breakout (trend) and 1d RSI mean reversion (extremes).
# Long when: price breaks above 4h Donchian upper channel AND 1d RSI < 30 (oversold).
# Short when: price breaks below 4h Donchian lower channel AND 1d RSI > 70 (overbought).
# Exit when price crosses back through the Donchian midpoint.
# Uses 4h for trend direction and structure, 1d for overextension filter.
# Session filter (08-20 UTC) reduces noise trades. Position size: 0.20.
# Target: 20-40 trades/year to minimize fee drag and improve generalization.

name = "1h_DonchianBreakout_4hTrend_1dRSI_Extreme"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 4h data for Donchian channel (trend structure)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Donchian channel (20-period) on 4h
    donchian_high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_mid_20 = (donchian_high_20 + donchian_low_20) / 2.0
    
    # Align Donchian levels to 1h
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_20)
    donchian_mid_20_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid_20)
    
    # Load 1d data for RSI (mean reversion filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # RSI (14-period) on 1d close
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_14 = 100 - (100 / (1 + rs))
    
    # Align RSI to 1h
    rsi_14_aligned = align_htf_to_ltf(prices, df_1d, rsi_14)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Sufficient warmup
    
    for i in range(start_idx, n):
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if np.isnan(donchian_high_20_aligned[i]) or np.isnan(donchian_low_20_aligned[i]) or \
           np.isnan(donchian_mid_20_aligned[i]) or np.isnan(rsi_14_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above 4h Donchian high AND 1d RSI < 30 (oversold)
            long_condition = (close[i] > donchian_high_20_aligned[i]) and (rsi_14_aligned[i] < 30)
            # Short: break below 4h Donchian low AND 1d RSI > 70 (overbought)
            short_condition = (close[i] < donchian_low_20_aligned[i]) and (rsi_14_aligned[i] > 70)
            
            if long_condition:
                signals[i] = 0.20
                position = 1
            elif short_condition:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price crosses below Donchian midpoint
            if close[i] < donchian_mid_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price crosses above Donchian midpoint
            if close[i] > donchian_mid_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals