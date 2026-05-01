#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly trend filter (price > weekly EMA50) and volume confirmation (volume > 1.8x 6h median).
# Long when price breaks above 6h Donchian upper band AND price > weekly EMA50 AND volume > 1.8x 6h volume median.
# Short when price breaks below 6h Donchian lower band AND price < weekly EMA50 AND volume > 1.8x 6h volume median.
# Uses discrete sizing 0.25. ATR(14) stoploss: signal→0 when price moves against position by 2.0*ATR.
# Weekly EMA50 provides strong trend filter that works in both bull and bear markets, Donchian breakouts capture momentum,
# volume confirmation filters false breakouts. Target: 12-25 trades/year on 6h timeframe (50-100 total over 4 years).

name = "6h_Donchian20_Breakout_WeeklyEMA50_Volume_v1"
timeframe = "6h"
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
    
    # Calculate 6h Donchian(20) bands from previous 6h bar
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    # Donchian bands: based on previous 6h bar's high/low over 20 periods
    prev_high_20 = pd.Series(df_6h['high'].values).rolling(window=20, min_periods=20).max().values
    prev_low_20 = pd.Series(df_6h['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian bands to 6h timeframe (already aligned to previous bar close)
    donchian_upper = align_htf_to_ltf(prices, df_6h, prev_high_20)
    donchian_lower = align_htf_to_ltf(prices, df_6h, prev_low_20)
    
    # Calculate weekly EMA50 trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 6h volume median (20-period for stability)
    vol_median_6h = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for ATR, EMA, Donchian, and volume
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(vol_median_6h[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 1.8x 6h volume median
        if vol_median_6h[i] <= 0 or np.isnan(vol_median_6h[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_6h[i] * 1.8)
        
        # Trend filter: price vs weekly EMA50
        uptrend = curr_close > ema_50_1w_aligned[i]
        downtrend = curr_close < ema_50_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Break above Donchian upper band AND uptrend AND volume confirmation
            if (curr_high > donchian_upper[i] and 
                uptrend and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Break below Donchian lower band AND downtrend AND volume confirmation
            elif (curr_low < donchian_lower[i] and 
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
            # Exit: price breaks below Donchian lower band OR trend turns down
            elif (curr_low < donchian_lower[i]) or (not uptrend):
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
            # Exit: price breaks above Donchian upper band OR trend turns up
            elif (curr_high > donchian_upper[i]) or (not downtrend):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals