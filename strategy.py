#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d EMA50 trend + volume spike + ATR stoploss.
# Long when price breaks above Donchian upper band (20-period high) AND price > 1d EMA50 AND volume > 1.8x 4h volume median.
# Short when price breaks below Donchian lower band (20-period low) AND price < 1d EMA50 AND volume > 1.8x 4h volume median.
# Uses discrete sizing 0.25. ATR(14) stoploss: signal→0 when price moves against position by 2.0*ATR.
# Target: 20-35 trades/year on 4h timeframe (80-140 total over 4 years) to minimize fee drag.
# Donchian channels provide objective price structure; EMA50 filters for higher-timeframe trend.
# Volume spike threshold set to 1.8x median to capture genuine momentum without excessive trades.

name = "4h_Donchian20_Breakout_1dEMA50_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian channels (20-period) from 4h data
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d EMA50 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 4h volume median (50-period for stability)
    vol_median_4h = pd.Series(volume).rolling(window=50, min_periods=50).median().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for ATR, Donchian, EMA, and volume
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_median_4h[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 1.8x 4h volume median
        if vol_median_4h[i] <= 0 or np.isnan(vol_median_4h[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_4h[i] * 1.8)
        
        # Trend filter: price vs 1d EMA50
        uptrend = curr_close > ema_50_1d_aligned[i]
        downtrend = curr_close < ema_50_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Break above Donchian high AND uptrend AND volume confirmation
            if (curr_high > donchian_high[i] and 
                uptrend and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Break below Donchian low AND downtrend AND volume confirmation
            elif (curr_low < donchian_low[i] and 
                  downtrend and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Stoploss: price moves against position by 2.0*ATR
            if curr_close < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks below Donchian low OR trend turns down
            elif (curr_low < donchian_low[i]) or (not uptrend):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: price moves against position by 2.0*ATR
            if curr_close > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks above Donchian high OR trend turns up
            elif (curr_high > donchian_high[i]) or (not downtrend):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals