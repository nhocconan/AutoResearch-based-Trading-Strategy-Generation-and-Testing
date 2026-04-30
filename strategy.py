#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using Donchian(20) breakout + 12h EMA(50) trend + volume confirmation
# Donchian breakouts capture institutional order flow at key structural levels.
# 12h EMA(50) ensures alignment with medium-term trend to avoid counter-trend trades.
# Volume confirmation filters false breakouts. Designed for low trade frequency (<25/year)
# to minimize fee drag in both bull and bear markets. Uses discrete position sizing.

name = "4h_Donchian20_Breakout_12hTrend_VolumeConfirm_v1"
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
    
    # Load 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    close_12h_s = pd.Series(close_12h)
    ema_50_12h = close_12h_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate ATR(14) for dynamic stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 50  # warmup for EMA(50)
    
    for i in range(start_idx, n):
        # Donchian(20) breakout levels
        lookback = min(20, i)
        donch_high = np.max(high[i-lookback:i])
        donch_low = np.min(low[i-lookback:i])
        
        # Volume confirmation: volume > 1.8x 20-period average
        vol_ma_20 = np.mean(volume[max(0, i-20):i])
        volume_spike = volume[i] > (1.8 * vol_ma_20)
        
        curr_close = close[i]
        curr_ema = ema_50_12h_aligned[i]
        curr_atr = atr[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and trend alignment
            if volume_spike:
                # Bullish entry: price breaks above Donchian high with 12h uptrend
                if curr_close > donch_high and curr_close > curr_ema:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below Donchian low with 12h downtrend
                elif curr_close < donch_low and curr_close < curr_ema:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.0 * ATR below entry price OR price breaks Donchian low
            if curr_close < entry_price - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close < donch_low:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches 1.5x ATR above entry
            elif curr_close >= entry_price + 1.5 * curr_atr:
                signals[i] = 0.10  # reduce position
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2.0 * ATR above entry price OR price breaks Donchian high
            if curr_close > entry_price + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close > donch_high:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches 1.5x ATR below entry
            elif curr_close <= entry_price - 1.5 * curr_atr:
                signals[i] = -0.10  # reduce position
            else:
                signals[i] = -0.25
    
    return signals