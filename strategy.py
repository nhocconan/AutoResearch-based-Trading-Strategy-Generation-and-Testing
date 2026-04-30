#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d EMA34 trend filter + volume spike confirmation + ATR-based stoploss
# Long when price breaks above 20-period Donchian high in uptrend (1d close > 1d EMA34) with volume spike (>1.8x average)
# Short when price breaks below 20-period Donchian low in downtrend (1d close < 1d EMA34) with volume spike
# Designed for low trade frequency (~20-50/year on 4h) to minimize fee drag while capturing strong directional moves
# Uses moderate volume confirmation (>1.8x average) and trend filter to reduce false breakouts
# Stoploss at 2.0 * ATR, take profit at 3.0 * ATR for asymmetric risk-reward (1:1.5)
# Works in bull markets via breakout continuation and in bear markets via fade of false breakouts at Donchian levels
# Focus on BTC/ETH as primary targets

name = "4h_Donchian20_Breakout_1dEMA34_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate ATR(14) for dynamic stoploss on 4h
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 20-period Donchian channels on 4h
    lookback = 20
    donchian_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donchian_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(lookback, 34)  # warmup for Donchian and EMA
    
    for i in range(start_idx, n):
        # Volume confirmation: volume > 1.8x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        elif i > 0:
            vol_ma_20 = np.mean(volume[:i])
        else:
            vol_ma_20 = 0
        volume_spike = volume[i] > (1.8 * vol_ma_20) if i > 0 else False
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_atr = atr[i]
        curr_dc_high = donchian_high[i]
        curr_dc_low = donchian_low[i]
        curr_ema = ema_34_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and trend alignment
            if volume_spike:
                # Bullish entry: price breaks above 20-period Donchian high with 1d uptrend (close > EMA34)
                if curr_close > curr_dc_high and curr_close > curr_ema:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below 20-period Donchian low with 1d downtrend (close < EMA34)
                elif curr_close < curr_dc_low and curr_close < curr_ema:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.0 * ATR below entry price
            if curr_close < entry_price - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches 3.0x ATR above entry
            elif curr_close > entry_price + 3.0 * curr_atr:
                signals[i] = 0.0  # full exit
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2.0 * ATR above entry price
            if curr_close > entry_price + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches 3.0x ATR below entry
            elif curr_close < entry_price - 3.0 * curr_atr:
                signals[i] = 0.0  # full exit
            else:
                signals[i] = -0.25
    
    return signals