#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_1dTrend_VolumeSpike_ATRStop
Hypothesis: On 12h timeframe, Donchian(20) breakouts filtered by 1d EMA50 trend and volume spikes capture strong momentum moves with controlled frequency. Long when price breaks above Donchian upper band in bullish 1d trend with volume confirmation; short when price breaks below lower band in bearish 1d trend with volume confirmation. Uses ATR-based stoploss and discrete sizing (±0.25) to target 12-37 trades/year. Works in bull markets via trend-following breaks and in bear markets via short breaks during downtrends, avoiding sideways chop via volume and trend filters.
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
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for higher-timeframe trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Donchian channels (20-period) on 12h
    period = 20
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # ATR (14) for stoploss and volume normalization
    tr1 = pd.Series(high).rolling(window=2).max().values - pd.Series(low).rolling(window=2).min().values
    tr2 = abs(pd.Series(high).rolling(window=2).max().values - pd.Series(close).shift(1).rolling(window=2).min().values)
    tr3 = abs(pd.Series(low).rolling(window=2).min().values - pd.Series(close).shift(1).rolling(window=2).max().values)
    tr = np.maximum.reduce([tr1, tr2, tr3])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume spike: volume > 1.5 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Warmup: max of Donchian(20), ATR(14), volume MA(20), 1d EMA50 alignment
    start_idx = max(period, 14, 20) + 4  # +4 to ensure 1d bar completion (12h -> 1d: 2 bars per day, but use 4 for safety)
    
    for i in range(start_idx, n):
        # Skip if any data not ready (NaN from calculation)
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(atr[i]) or np.isnan(ema_50_1d_aligned[i])):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        close_val = close[i]
        upper_band = highest_high[i]
        lower_band = lowest_low[i]
        ema_50_val = ema_50_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine 1d trend: bullish if price > EMA50, bearish if price < EMA50
        bullish_1d = close_val > ema_50_val
        bearish_1d = close_val < ema_50_val
        
        # Entry conditions: Donchian breakout in direction of 1d trend with volume confirmation
        long_entry = (close_val > upper_band) and bullish_1d and vol_spike
        short_entry = (close_val < lower_band) and bearish_1d and vol_spike
        
        # ATR-based stoploss
        long_stop = position == 1 and close_val < (entry_price - 2.5 * atr[i]) if position == 1 and 'entry_price' in locals() else False
        short_stop = position == -1 and close_val > (entry_price + 2.5 * atr[i]) if position == -1 and 'entry_price' in locals() else False
        
        # Exit on stoploss or opposite breakout
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
            entry_price = close_val  # approximate entry price for stop
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
            entry_price = close_val  # approximate entry price for stop
        elif position == 1 and (close_val < (entry_price - 2.5 * atr[i]) or not bullish_1d):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close_val > (entry_price + 2.5 * atr[i]) or not bearish_1d):
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "12h_Donchian20_Breakout_1dTrend_VolumeSpike_ATRStop"
timeframe = "12h"
leverage = 1.0