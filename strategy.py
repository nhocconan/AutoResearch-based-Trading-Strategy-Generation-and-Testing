#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume spike (>2.0x 30-period average)
# Donchian breakout captures strong momentum moves, 1d EMA34 ensures alignment with daily trend,
# volume confirmation filters false breakouts. Works in bull/bear: trend filter avoids whipsaws,
# volume confirms institutional participation. Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.

name = "4h_Donchian20_VolumeSpike_1dEMA34_Trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    high_ma_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 2.0x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_confirm = volume > (2.0 * vol_ma_30)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(20, 30, 34)  # warmup for Donchian, volume MA, 1d EMA
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(high_ma_20[i]) or np.isnan(low_ma_20[i]) or 
            np.isnan(vol_ma_30[i]) or np.isnan(ema_34_aligned[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high_ma = high_ma_20[i]
        curr_low_ma = low_ma_20[i]
        curr_volume_confirm = volume_confirm[i]
        curr_ema_34 = ema_34_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation and trend filter
            if curr_volume_confirm:
                # Bullish breakout: price breaks above upper Donchian with price above 1d EMA34
                if curr_close > curr_high_ma and curr_close > curr_ema_34:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish breakout: price breaks below lower Donchian with price below 1d EMA34
                elif curr_close < curr_low_ma and curr_close < curr_ema_34:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit when price breaks below lower Donchian (trend reversal)
            if curr_close < curr_low_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when price breaks above upper Donchian (trend reversal)
            if curr_close > curr_high_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals