#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation.
# Long when price breaks above Donchian upper channel AND close > 1d EMA34 AND volume > 1.5x 20-period MA.
# Short when price breaks below Donchian lower channel AND close < 1d EMA34 AND volume > 1.5x 20-period MA.
# Uses discrete sizing 0.25 with ATR-based stoploss (signal→0 when price moves against position by 2.0*ATR).
# Target: 75-200 total trades over 4 years (19-50/year) with Sharpe > 0 on BTC/ETH/SOL.

name = "4h_Donchian20_1dEMA34_Volume_ATR"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Donchian channels (20-period) on 4h
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Calculate ATR(14) for stoploss
    tr1 = pd.Series(high - low).values
    tr2 = pd.Series(np.abs(high - np.roll(close, 1))).values
    tr3 = pd.Series(np.abs(low - np.roll(close, 1))).values
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(lookback, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_34_1d_aligned[i]
        upper_channel = highest_high[i]
        lower_channel = lowest_low[i]
        atr_val = atr[i]
        
        # Volume regime: current 4h volume > 1.5x 20-period MA
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
            volume_spike = volume[i] > (1.5 * vol_ma_20)
        else:
            volume_spike = False
        
        # Stoploss logic
        if position == 1 and close_val < entry_price - 2.0 * atr_val:
            signals[i] = 0.0
            position = 0
            continue
        elif position == -1 and close_val > entry_price + 2.0 * atr_val:
            signals[i] = 0.0
            position = 0
            continue
        
        # Entry logic
        if position == 0:
            if close_val > upper_channel and close_val > ema_trend and volume_spike:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif close_val < lower_channel and close_val < ema_trend and volume_spike:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
        elif position == 1:
            # Long exit: price breaks below lower channel OR trend reversal
            if close_val < lower_channel or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above upper channel OR trend reversal
            if close_val > upper_channel or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals