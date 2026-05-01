#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
# Long when price breaks above Donchian upper (20) AND close > 1d EMA50 AND volume > 1.3x 20-bar average.
# Short when price breaks below Donchian lower (20) AND close < 1d EMA50 AND volume > 1.3x 20-bar average.
# Uses discrete sizing 0.25 to manage drawdown. ATR-based stoploss exits when price moves against position by 2.5x ATR.
# Target: 80-180 total trades over 4 years (20-45/year) for 4h timeframe.
# Donchian channels provide robust structure that works in both bull (breakouts up) and bear (breakdowns down).
# 1d EMA50 offers smooth trend alignment that avoids whipsaw in ranging markets.
# Volume confirmation ensures only high-conviction breakouts are traded.

name = "4h_Donchian20_1dEMA50_Trend_VolumeConfirm_v1"
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
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 calculation
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Donchian channels (20-period) on 4h data
    lookback = 20
    donchian_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donchian_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # ATR (14) for stoploss calculation
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: current 4h volume > 1.3x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    start_idx = max(lookback, 20, 14) + 5  # warmup for all indicators
    
    for i in range(start_idx, n):
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma[i]
        curr_atr = atr[i]
        
        if curr_vol_ma <= 0 or curr_atr <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = curr_vol > (curr_vol_ma * 1.3)
        
        # Donchian breakout signals
        breakout_up = curr_high > donchian_high[i]  # break above upper band
        breakout_down = curr_low < donchian_low[i]  # break below lower band
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: breakout above Donchian upper AND close > 1d EMA50 AND volume confirmation
            if (breakout_up and 
                curr_close > ema_50_aligned[i] and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: breakout below Donchian lower AND close < 1d EMA50 AND volume confirmation
            elif (breakout_down and 
                  curr_close < ema_50_aligned[i] and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # ATR-based stoploss: exit if price drops below entry - 2.5*ATR
            if curr_low < (entry_price - 2.5 * curr_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # ATR-based stoploss: exit if price rises above entry + 2.5*ATR
            if curr_high > (entry_price + 2.5 * curr_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals