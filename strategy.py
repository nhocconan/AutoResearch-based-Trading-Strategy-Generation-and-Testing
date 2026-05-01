#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume spike confirmation.
# Long when price breaks above Donchian upper (20) AND price > 1d EMA50 AND volume > 2.0x 4h volume median.
# Short when price breaks below Donchian lower (20) AND price < 1d EMA50 AND volume > 2.0x 4h volume median.
# Uses discrete sizing 0.25. ATR(14) stoploss: signal→0 when price moves against position by 2.5*ATR.
# Donchian channels provide robust price structure, 1d EMA50 offers smooth trend filter, volume spike confirms momentum.
# Target: 20-30 trades/year on 4h timeframe (80-120 total over 4 years) to minimize fee drag and ensure BTC/ETH efficacy.

name = "4h_Donchian20_Breakout_1dEMA50_Volume_v2"
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
    
    # Calculate 4h Donchian channels (20-period) from previous 4h bar
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Donchian levels: based on previous 4h bar's high/low (20-period)
    high_20 = pd.Series(df_4h['high'].values).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_4h['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe (already aligned to previous bar close)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, low_20)
    
    # Calculate 1d EMA50 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 4h volume median (20-period for stability)
    vol_median_4h = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for ATR, EMA, and volume
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_median_4h[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 2.0x 4h volume median
        if vol_median_4h[i] <= 0 or np.isnan(vol_median_4h[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_4h[i] * 2.0)
        
        # Trend filter: price vs 1d EMA50
        uptrend = curr_close > ema_50_1d_aligned[i]
        downtrend = curr_close < ema_50_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Break above Donchian high AND uptrend AND volume confirmation
            if (curr_high > donchian_high_aligned[i] and 
                uptrend and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Break below Donchian low AND downtrend AND volume confirmation
            elif (curr_low < donchian_low_aligned[i] and 
                  downtrend and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Stoploss: price moves against position by 2.5*ATR
            if curr_close < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks below Donchian low OR trend turns down
            elif (curr_low < donchian_low_aligned[i]) or (not uptrend):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: price moves against position by 2.5*ATR
            if curr_close > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks above Donchian high OR trend turns up
            elif (curr_high > donchian_high_aligned[i]) or (not downtrend):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals