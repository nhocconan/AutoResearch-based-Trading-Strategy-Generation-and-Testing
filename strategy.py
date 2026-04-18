#!/usr/bin/env python3
"""
4h_PriceChannel_Breakout_1dTrend_VolumeSpike
Hypothesis: Price channels (donchian-like) based on prior 12h high/low act as support/resistance.
Breakouts above/below these levels with volume spike and daily EMA(34) trend filter capture momentum.
Designed to work in both bull and bear by requiring trend alignment (EMA) and volume confirmation.
Target: 10-20 trades/year on 4h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for channel calculation (prior period high/low)
    df_12h = get_htf_data(prices, '12h')
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12-period high/low on 12h timeframe (for channel)
    high_12h = df_12h['high']
    low_12h = df_12h['low']
    
    # Upper channel: 12h high, lower channel: 12h low
    upper_channel = high_12h
    lower_channel = low_12h
    
    # Shift by 1 to use only completed 12h periods (no look-ahead)
    upper_prev = upper_channel.shift(1).values
    lower_prev = lower_channel.shift(1).values
    
    # Align to 4h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_12h, upper_prev)
    lower_aligned = align_htf_to_ltf(prices, df_12h, lower_prev)
    
    # Daily EMA(34) for trend filter
    close_1d = df_1d['close']
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # ATR for volatility filter (14-period)
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volatility filter: only trade when ATR > 20-period average (avoid chop)
    atr_ma = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    volatility_filter = atr > atr_ma
    
    # Volume spike: 3.0x 20-period average on 4h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (3.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    bars_since_entry = 0  # track holding period
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(upper_aligned[i]) or 
            np.isnan(lower_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(atr[i]) or
            np.isnan(atr_ma[i])):
            signals[i] = 0.0
            bars_since_entry = 0
            continue
        
        price = close[i]
        upper_val = upper_aligned[i]
        lower_val = lower_aligned[i]
        ema_trend = ema_34_1d_aligned[i]
        vol_filter = volatility_filter[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            bars_since_entry = 0
            # Long: break above upper channel with volume spike, price above daily EMA, and sufficient volatility
            if price > upper_val and vol_spike and price > ema_trend and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: break below lower channel with volume spike, price below daily EMA, and sufficient volatility
            elif price < lower_val and vol_spike and price < ema_trend and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Minimum holding period: 4 bars (16 hours for 4h)
            if bars_since_entry < 4:
                signals[i] = 0.25
                bars_since_entry += 1
            else:
                signals[i] = 0.25
                # Exit: price returns to lower channel or breaks below daily EMA
                if price <= lower_val or price < ema_trend:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
        
        elif position == -1:
            # Minimum holding period: 4 bars (16 hours for 4h)
            if bars_since_entry < 4:
                signals[i] = -0.25
                bars_since_entry += 1
            else:
                signals[i] = -0.25
                # Exit: price returns to upper channel or breaks above daily EMA
                if price >= upper_val or price > ema_trend:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
    
    return signals

name = "4h_PriceChannel_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0