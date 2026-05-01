#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w trend filter and volume confirmation.
# Long when price breaks above Donchian(20) high AND price > 1w EMA50 AND volume > 1.5x 12h volume median.
# Short when price breaks below Donchian(20) low AND price < 1w EMA50 AND volume > 1.5x 12h volume median.
# Uses discrete sizing 0.25. ATR(14) stoploss: signal→0 when price moves against position by 2.0*ATR.
# Donchian channels provide clear breakout levels. 1w EMA50 filters for higher-timeframe trend.
# Volume confirmation ensures breakouts have momentum. Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years).

name = "12h_Donchian20_1wEMA50_Volume_v1"
timeframe = "12h"
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
    
    # Calculate 12h Donchian(20)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1w EMA50 trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 12h volume median (20-period for stability)
    vol_median_12h = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for ATR, Donchian, EMA, and volume
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_median_12h[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Donchian breakout conditions
        breakout_high = curr_close > donchian_high[i-1]  # break above previous period's high
        breakout_low = curr_close < donchian_low[i-1]    # break below previous period's low
        
        # Trend filter: price vs 1w EMA50
        uptrend = curr_close > ema_50_1w_aligned[i]
        downtrend = curr_close < ema_50_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 12h volume median
        if vol_median_12h[i] <= 0 or np.isnan(vol_median_12h[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_12h[i] * 1.5)
        
        if position == 0:  # Flat - look for new entries
            # Long: Donchian breakout high AND uptrend AND volume spike
            if breakout_high and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Donchian breakout low AND downtrend AND volume spike
            elif breakout_low and downtrend and volume_confirm:
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
            elif curr_close < donchian_low[i] or not uptrend:
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
            elif curr_close > donchian_high[i] or not downtrend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals