#1h_4h_1d_Trend_Momentum_Volume
name = "1h_4h_1d_Trend_Momentum_Volume"
timeframe = "1h"
leverage = 1.0

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
    
    # Load 4h data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 4h EMA20 for trend filter
    ema20_4h = pd.Series(df_4h['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # 4h RSI for momentum filter
    delta_4h = pd.Series(df_4h['close']).diff()
    gain_4h = (delta_4h.where(delta_4h > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss_4h = (-delta_4h.where(delta_4h < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs_4h = gain_4h / loss_4h
    rsi14_4h = 100 - (100 / (1 + rs_4h))
    rsi14_4h = rsi14_4h.fillna(50).values  # Neutral when undefined
    rsi14_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi14_4h)
    
    # 1d volume spike: > 1.8x 20-period average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = vol_1d > 1.8 * vol_ma_1d
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(40, 20)  # Wait for indicators
    
    for i in range(start_idx, n):
        if not session_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if np.isnan(ema20_4h_aligned[i]) or np.isnan(rsi14_4h_aligned[i]) or np.isnan(vol_spike_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above 4h EMA20, RSI > 55, 1d volume spike
            if close[i] > ema20_4h_aligned[i] and rsi14_4h_aligned[i] > 55 and vol_spike_1d_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: Price below 4h EMA20, RSI < 45, 1d volume spike
            elif close[i] < ema20_4h_aligned[i] and rsi14_4h_aligned[i] < 45 and vol_spike_1d_aligned[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: Price below 4h EMA20 or RSI < 40
            if close[i] < ema20_4h_aligned[i] or rsi14_4h_aligned[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: Price above 4h EMA20 or RSI > 60
            if close[i] > ema20_4h_aligned[i] or rsi14_4h_aligned[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

# Hypothesis: 1h trend following with 4h EMA20 trend filter and 4h RSI momentum filter.
# Long when price > 4h EMA20, RSI > 55 (bullish momentum), and 1d volume spike confirms conviction.
# Short when price < 4h EMA20, RSI < 45 (bearish momentum), and 1d volume spike confirms conviction.
# Uses 4h timeframe for trend/momentum to avoid whipsaws, 1h for entry timing.
# 1d volume spike (>1.8x average) ensures institutional participation.
# Session filter (08-20 UTC) reduces noise during low-liquidity hours.
# Discrete 0.20 position size limits risk and controls drawdown.
# Target: 15-35 trades/year to minimize fee drag while capturing sustained moves in both bull and bear markets.