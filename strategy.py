#!/usr/bin/env python3
"""
1h 4h/1d Trend + Volume Spike Strategy
Hypothesis: In strong trends (4h EMA200) with volume spikes (>1.5x 20-period avg),
price tends to continue in trend direction for 1-3 hours. Use 1d ADX>25 to filter
weak trend days. Designed for low frequency: ~20-40 trades/year.
Works in bull/bear by following trend direction. Entry: price > 4h EMA20 (long) or < (short)
with volume spike and 1d ADX>25. Exit: trend reversal or volume dry-up.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA20 trend
    df_4h = get_htf_data(prices, '4h')
    ema_20_4h = pd.Series(df_4h['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    # Calculate ADX(14) on daily
    plus_dm = np.diff(df_1d['high'], prepend=df_1d['high'][0])
    minus_dm = np.diff(df_1d['low'], prepend=df_1d['low'][0])
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    tr = np.maximum.reduce([
        np.abs(np.diff(df_1d['high'], prepend=df_1d['high'][0])),
        np.abs(np.diff(df_1d['low'], prepend=df_1d['low'][0])),
        np.abs(df_1d['high'] - df_1d['low'])
    ])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr_14
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr_14
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx_14 = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Volume spike detection on 1h (1.5x 20-period avg)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 50  # need enough history for ADX
    
    for i in range(start_idx, n):
        if (np.isnan(ema_20_4h_aligned[i]) or 
            np.isnan(adx_14_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_20_4h = ema_20_4h_aligned[i]
        adx = adx_14_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price above 4h EMA20, strong trend (ADX>25), volume spike
            if price > ema_20_4h and adx > 25 and vol_spike:
                signals[i] = 0.20
                position = 1
            # Short: price below 4h EMA20, strong trend (ADX>25), volume spike
            elif price < ema_20_4h and adx > 25 and vol_spike:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long position management
            signals[i] = 0.20
            # Exit: trend weakens (ADX<20) or price crosses below EMA20
            if adx < 20 or price < ema_20_4h:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position management
            signals[i] = -0.20
            # Exit: trend weakens (ADX<20) or price crosses above EMA20
            if adx < 20 or price > ema_20_4h:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Trend_VolumeSpike_ADXFilter"
timeframe = "1h"
leverage = 1.0