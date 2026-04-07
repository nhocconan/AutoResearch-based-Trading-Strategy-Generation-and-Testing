#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly trend filter and volume confirmation
# Long when price breaks above 6h Donchian high (20-period) and 1w close > 1w EMA10 (bullish)
# Short when price breaks below 6h Donchian low (20-period) and 1w close < 1w EMA10 (bearish)
# Exit on opposite Donchian break or stoploss at 2.5 * ATR(14)
# Position size: 0.25 (25% of capital)
# Uses weekly trend filter to avoid counter-trend trades in ranging markets
# Target: 50-150 total trades over 4 years (12-37/year) with low churn

name = "6h_donchian20_1w_trend_vol_v1"
timeframe = "6h"
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
    
    # 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1-week EMA(10) for trend filter
    close_1w = df_1w['close'].values
    ema_10_1w = pd.Series(close_1w).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema_10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_10_1w)
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume moving average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(ema_10_1w_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i]):
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
            # Exit: price breaks below Donchian low (20-period)
            elif close[i] < low[i-20]:
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
            # Exit: price breaks above Donchian high (20-period)
            elif close[i] > high[i-20]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Calculate Donchian channels from 20 periods ago
            donch_high = high[i-20]
            donch_low = low[i-20]
            
            # Volume confirmation: current volume > 20-period average
            vol_confirm = volume[i] > vol_ma[i]
            
            # Trend filter: 1w EMA10
            uptrend = close[i] > ema_10_1w_aligned[i]
            downtrend = close[i] < ema_10_1w_aligned[i]
            
            # Long: price breaks above Donchian high with volume and uptrend
            if close[i] > donch_high and vol_confirm and uptrend:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below Donchian low with volume and downtrend
            elif close[i] < donch_low and vol_confirm and downtrend:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals