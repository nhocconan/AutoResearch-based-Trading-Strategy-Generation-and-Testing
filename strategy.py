#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d trend filter and volume confirmation
# Long when price breaks above Donchian(20) upper band and 1d close > 1d EMA50 and volume > 1.5x average
# Short when price breaks below Donchian(20) lower band and 1d close < 1d EMA50 and volume > 1.5x average
# Exit when price crosses opposite Donchian band or stoploss at 2.5 * ATR
# Position size: 0.25 (25% of capital)
# Target: 50-150 total trades over 4 years (12-37/year)

name = "12h_donchian20_1d_trend_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1-day EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Average volume for confirmation (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if required data not available
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.5 * ATR
            if close[i] < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses below Donchian lower band (20-period)
            elif close[i] < np.min(low[i-20:i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.5 * ATR
            if close[i] > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses above Donchian upper band (20-period)
            elif close[i] > np.max(high[i-20:i]):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Calculate Donchian bands from previous bar (12h)
            # Using high and low from 20 bars ago (previous 20 periods)
            donchian_upper = np.max(high[i-20:i])
            donchian_lower = np.min(low[i-20:i])
            
            # Trend filter: 1d EMA50
            uptrend = close[i] > ema_50_1d_aligned[i]
            downtrend = close[i] < ema_50_1d_aligned[i]
            
            # Volume confirmation: current volume > 1.5x average
            vol_confirm = volume[i] > 1.5 * vol_ma[i]
            
            # Long: price breaks above Donchian upper band in uptrend with volume
            if close[i] > donchian_upper and close[i-1] <= donchian_upper and uptrend and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian lower band in downtrend with volume
            elif close[i] < donchian_lower and close[i-1] >= donchian_lower and downtrend and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals