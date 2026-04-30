#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d ATR-based volatility breakout with 1w EMA34 trend filter
# Long when price breaks above 1d high + 0.5*ATR(14) in 1w uptrend with volume confirmation
# Short when price breaks below 1d low - 0.5*ATR(14) in 1w downtrend with volume confirmation
# Designed for low trade frequency (~12-37/year on 12h) to minimize fee drag while capturing strong directional moves
# Works in bull markets via breakout continuation and in bear markets via fade of false breakouts
# Focus on BTC/ETH as primary targets with SOL as secondary

name = "12h_1dATRBreakout_1wEMA34_VolumeConfirm_v1"
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
    
    # Load 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for volatility measurement
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1d high and low for breakout levels
    # Breakout levels: high + 0.5*ATR, low - 0.5*ATR
    breakout_up = high_1d + 0.5 * atr_1d
    breakout_down = low_1d - 0.5 * atr_1d
    
    # Align 1d breakout levels to 12h timeframe (wait for 1d bar to close)
    breakout_up_aligned = align_htf_to_ltf(prices, df_1d, breakout_up)
    breakout_down_aligned = align_htf_to_ltf(prices, df_1d, breakout_down)
    
    # Calculate 1w EMA(34) for trend filter
    close_1w = df_1w['close'].values
    ema_34 = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34)
    
    # Calculate 12h ATR(14) for dynamic stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 40  # warmup for EMA(34) and ATR
    
    for i in range(start_idx, n):
        # Volume confirmation: volume > 1.8x 40-period average
        if i >= 40:
            vol_ma_40 = np.mean(volume[i-40:i])
        elif i > 0:
            vol_ma_40 = np.mean(volume[:i])
        else:
            vol_ma_40 = 0
        volume_spike = volume[i] > (1.8 * vol_ma_40) if i > 0 else False
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_atr = atr[i]
        curr_breakout_up = breakout_up_aligned[i]
        curr_breakout_down = breakout_down_aligned[i]
        curr_ema = ema_34_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and trend alignment
            if volume_spike:
                # Bullish entry: price breaks above 1d breakout up with 1w uptrend (close > EMA34)
                if curr_close > curr_breakout_up and curr_close > curr_ema:
                    signals[i] = 0.30
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below 1d breakout down with 1w downtrend (close < EMA34)
                elif curr_close < curr_breakout_down and curr_close < curr_ema:
                    signals[i] = -0.30
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.0 * ATR below entry price OR price breaks 1d breakout down (reversal signal)
            if curr_close < entry_price - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close < curr_breakout_down:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches 3.0x ATR above entry
            elif curr_close > entry_price + 3.0 * curr_atr:
                signals[i] = 0.0  # full exit
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Stoploss: 2.0 * ATR above entry price OR price breaks 1d breakout up (reversal signal)
            if curr_close > entry_price + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close > curr_breakout_up:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches 3.0x ATR below entry
            elif curr_close < entry_price - 3.0 * curr_atr:
                signals[i] = 0.0  # full exit
            else:
                signals[i] = -0.30
    
    return signals