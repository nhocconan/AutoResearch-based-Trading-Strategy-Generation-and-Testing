#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Williams %R extremes with 12h EMA200 trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions on daily timeframe.
# Long when Williams %R < -80 (oversold) and price > 12h EMA200 (uptrend) with volume spike.
# Short when Williams %R > -20 (overbought) and price < 12h EMA200 (downtrend) with volume spike.
# Uses 1d HTF for Williams %R to avoid intraday noise and 12h EMA200 for strong trend filter.
# Volume spike (>2.0x average) confirms institutional participation.
# Designed for low trade frequency (~15-30/year on 4h) to minimize fee drag while capturing reversals at extremes.
# Works in bull markets via buying oversold dips in uptrend and in bear markets via selling overbought rallies in downtrend.
# Focus on BTC/ETH as primary targets.

name = "4h_1dWilliamsR_Extreme_12hEMA200_VolumeConfirm_v1"
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
    
    # Load 1d data ONCE before loop for Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Williams %R(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid division by zero
    
    # Align 1d Williams %R to 4h timeframe (wait for 1d bar to close)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Load 12h data ONCE before loop for EMA200 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA(200)
    close_12h = df_12h['close'].values
    ema_200_12h = pd.Series(close_12h).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 12h EMA200 to 4h timeframe (wait for 12h bar to close)
    ema_200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_200_12h)
    
    # Calculate ATR(14) for dynamic stoploss on 4h
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 200  # warmup for EMA(200)
    
    for i in range(start_idx, n):
        # Volume confirmation: volume > 2.0x 50-period average
        if i >= 50:
            vol_ma_50 = np.mean(volume[i-50:i])
        elif i > 0:
            vol_ma_50 = np.mean(volume[:i])
        else:
            vol_ma_50 = 0
        volume_spike = volume[i] > (2.0 * vol_ma_50) if i > 0 else False
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_atr = atr[i]
        curr_williams_r = williams_r_aligned[i]
        curr_ema = ema_200_12h_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if volume_spike:
                # Bullish entry: Williams %R oversold (< -80) and price > 12h EMA200 (uptrend)
                if curr_williams_r < -80 and curr_close > curr_ema:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: Williams %R overbought (> -20) and price < 12h EMA200 (downtrend)
                elif curr_williams_r > -20 and curr_close < curr_ema:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.5 * ATR below entry price OR Williams %R > -20 (overbought exit)
            if curr_close < entry_price - 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_williams_r > -20:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches 2.0x ATR above entry
            elif curr_close > entry_price + 2.0 * curr_atr:
                signals[i] = 0.0  # full exit
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2.5 * ATR above entry price OR Williams %R < -80 (oversold exit)
            if curr_close > entry_price + 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_williams_r < -80:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches 2.0x ATR below entry
            elif curr_close < entry_price - 2.0 * curr_atr:
                signals[i] = 0.0  # full exit
            else:
                signals[i] = -0.25
    
    return signals