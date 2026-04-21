#!/usr/bin/env python3
"""
4h_KeltnerChannel_Breakout_HTFTrend_ATRStop
Hypothesis: 4h price breakout beyond Keltner Channel (EMA20 ± 2*ATR(10)) filtered by 1d EMA50 trend.
Enter long when price closes above upper KC with 1d uptrend. Enter short when price closes below lower KC with 1d downtrend.
Exit on opposite KC touch or ATR(10) trailing stop (1.5*ATR). Uses volume confirmation to avoid false breakouts.
Designed for low trade frequency (target: 20-40 trades/year) to minimize fee drag in ranging/bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for trend filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 4h Keltner Channel: EMA20 ± 2*ATR(10) ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # EMA20 for KC middle
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # ATR(10) for KC width
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_10 = tr.rolling(window=10, min_periods=10).mean().values
    
    # KC bands
    kc_upper = ema_20 + 2.0 * atr_10
    kc_lower = ema_20 - 2.0 * atr_10
    
    # === 1d EMA50 for HTF trend filter ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === Volume confirmation: 20-period average ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(ema_20[i]) or np.isnan(kc_upper[i]) or np.isnan(kc_lower[i]) 
            or np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i]) 
            or np.isnan(atr_10[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Volume confirmation: current volume > 20-period average
            vol_confirm = volume[i] > vol_ma[i]
            
            # Long conditions: price > upper KC, 1d uptrend, volume spike
            long_breakout = price > kc_upper[i]
            long_trend = price > ema_50_1d_aligned[i]
            
            # Short conditions: price < lower KC, 1d downtrend, volume spike
            short_breakout = price < kc_lower[i]
            short_trend = price < ema_50_1d_aligned[i]
            
            # Entry logic
            if long_breakout and long_trend and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_breakout and short_trend and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 1.5 * atr_10[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price touches or crosses lower KC
            elif price <= kc_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 1.5 * atr_10[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price touches or crosses upper KC
            elif price >= kc_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_KeltnerChannel_Breakout_HTFTrend_ATRStop"
timeframe = "4h"
leverage = 1.0