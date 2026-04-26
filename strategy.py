#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeSpike_HTFTrend
Hypothesis: On 4h timeframe, Donchian(20) breakouts with volume confirmation and 1d EMA50 trend filter capture strong momentum moves while minimizing false breakouts. Long when price breaks above 20-period high with volume spike in bullish 1d trend; short when price breaks below 20-period low with volume spike in bearish 1d trend. Uses discrete sizing (±0.30) and ATR-based stoploss. Designed for 19-50 trades/year to avoid fee drag and work in both bull/bear markets by trading only with higher-timeframe trend.
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
    
    # Load 1d data ONCE before loop for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for higher-timeframe trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Donchian channels: 20-period high/low
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # ATR(14) for stoploss and position sizing reference
    tr1 = pd.Series(high - low).values
    tr2 = pd.Series(np.abs(high - np.roll(close, 1))).values
    tr3 = pd.Series(np.abs(low - np.roll(close, 1))).values
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.30
    
    # Warmup: max of calculations (20 for Donchian, 20 for volume MA, 14 for ATR)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready (NaN from calculation)
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr[i]) or
            np.isnan(vol_ma[i])):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        close_val = close[i]
        donch_high = donchian_high[i]
        donch_low = donchian_low[i]
        ema_50_val = ema_50_1d_aligned[i]
        atr_val = atr[i]
        vol_spike = volume_spike[i]
        
        # Determine 1d trend: bullish if price > EMA50, bearish if price < EMA50
        bullish_1d = close_val > ema_50_val
        bearish_1d = close_val < ema_50_val
        
        # Entry conditions: Donchian breakout with volume spike in direction of 1d trend
        long_entry = (close_val > donch_high) and bullish_1d and vol_spike
        short_entry = (close_val < donch_low) and bearish_1d and vol_spike
        
        # Stoploss conditions: ATR-based trailing stop
        long_stop = position == 1 and close_val <= (donch_high - 2.0 * atr_val)
        short_stop = position == -1 and close_val >= (donch_low + 2.0 * atr_val)
        
        # Update signals and position
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
        elif long_stop or short_stop:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "4h_Donchian20_Breakout_VolumeSpike_HTFTrend"
timeframe = "4h"
leverage = 1.0