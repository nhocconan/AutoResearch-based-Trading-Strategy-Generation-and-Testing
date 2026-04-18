#!/usr/bin/env python3
"""
1d_TrendFollow_Volume_Signal_Adaptive
Hypothesis: On daily timeframe, trend following with volume confirmation captures major moves while avoiding whipsaws.
In bull markets, buy on strong up days with volume; in bear markets, sell on strong down days with volume.
Uses adaptive thresholds based on volatility to maintain consistent signal frequency.
Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for trend and volatility
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Weekly data for trend filter (avoid counter-trend trades)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Daily 20-period EMA for trend
    ema_20 = pd.Series(df_1d['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_aligned = align_htf_to_ltf(prices, df_1d, ema_20)
    
    # Daily ATR for volatility normalization
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    atr = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    
    # Weekly trend filter: price above/below weekly EMA
    weekly_ema = pd.Series(df_1w['close']).ewm(span=10, adjust=False, min_periods=10).mean().values
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    # Daily price change normalized by ATR
    price_change = np.diff(close, prepend=close[0])
    price_change_norm = price_change / atr_aligned
    
    # Volume filter: >1.3x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 30  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema_20_aligned[i]) or np.isnan(weekly_ema_aligned[i]) or
            np.isnan(atr_aligned[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_trend = ema_20_aligned[i]
        weekly_trend = weekly_ema_aligned[i]
        vol_ok = volume_filter[i]
        price_norm = price_change_norm[i]
        
        if position == 0:
            # Long: strong up day with volume, aligned with weekly trend
            if price_norm > 0.8 and vol_ok and price > weekly_trend:
                signals[i] = 0.30
                position = 1
            # Short: strong down day with volume, aligned with weekly trend
            elif price_norm < -0.8 and vol_ok and price < weekly_trend:
                signals[i] = -0.30
                position = -1
        
        elif position == 1:
            # Exit: trend reversal or volatility drop
            if price < ema_trend or abs(price_norm) < 0.3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Exit: trend reversal or volatility drop
            if price > ema_trend or abs(price_norm) < 0.3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "1d_TrendFollow_Volume_Signal_Adaptive"
timeframe = "1d"
leverage = 1.0