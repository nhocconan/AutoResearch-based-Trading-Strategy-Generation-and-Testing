#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
# Long when price breaks above Donchian upper AND price > 12h EMA50 AND volume > 1.5x 4h volume average.
# Short when price breaks below Donchian lower AND price < 12h EMA50 AND volume > 1.5x 4h volume average.
# Uses discrete sizing 0.30. ATR(14) stoploss: signal→0 when price moves against position by 2.0*ATR.
# Donchian calculated from prior completed 4h bar to avoid look-ahead.
# Volume spike filters low-momentum signals. 12h EMA50 ensures trades only in established medium-term trends.
# Works in bull (breakouts with uptrend) and bear (breakdowns with downtrend).
# Target: 25-50 trades/year on 4h timeframe.

name = "4h_Donchian20_12hEMA50_Volume_v1"
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
    
    # Load 4h data ONCE before loop for Donchian and volume filters (primary timeframe data)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate Donchian(20) on 4h timeframe
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Donchian upper: 20-period rolling max of high
    donch_upper = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    # Donchian lower: 20-period rolling min of low
    donch_lower = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h volume average (20-period)
    vol_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align to 4h timeframe (no additional delay needed for Donchian as it's based on completed 4h bar)
    donch_upper_aligned = align_htf_to_ltf(prices, df_4h, donch_upper)
    donch_lower_aligned = align_htf_to_ltf(prices, df_4h, donch_lower)
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    # Load 12h data ONCE before loop for EMA50 trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for ATR, Donchian, volume, and EMA
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or 
            np.isnan(donch_upper_aligned[i]) or 
            np.isnan(donch_lower_aligned[i]) or 
            np.isnan(vol_ma_4h_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume spike: current volume > 1.5x 4h volume average
        if vol_ma_4h_aligned[i] <= 0 or np.isnan(vol_ma_4h_aligned[i]):
            volume_spike = False
        else:
            volume_spike = curr_volume > (vol_ma_4h_aligned[i] * 1.5)
        
        # Donchian breakout conditions
        breakout_up = curr_high > donch_upper_aligned[i]
        breakout_down = curr_low < donch_lower_aligned[i]
        
        # Trend filter: price vs 12h EMA50
        uptrend = curr_close > ema_50_12h_aligned[i]
        downtrend = curr_close < ema_50_12h_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Donchian breakout up AND volume spike AND uptrend
            if (breakout_up and 
                volume_spike and 
                uptrend):
                signals[i] = 0.30
                position = 1
                entry_price = curr_close
            # Short: Donchian breakout down AND volume spike AND downtrend
            elif (breakout_down and 
                  volume_spike and 
                  downtrend):
                signals[i] = -0.30
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
            # Exit: price re-enters Donchian channel OR trend reverses
            elif (curr_close < donch_upper_aligned[i]) or (not uptrend):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Stoploss: price moves against position by 2.0*ATR
            if curr_close > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price re-enters Donchian channel OR trend reverses
            elif (curr_close > donch_lower_aligned[i]) or (not downtrend):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.30
    
    return signals