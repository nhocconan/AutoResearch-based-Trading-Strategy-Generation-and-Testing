#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Williams %R extremes (oversold/overbought) with 1d EMA34 trend filter and volume confirmation
# Long when Williams %R < -80 (oversold) and price > 1d EMA34 (uptrend) with volume spike (>2.0x average)
# Short when Williams %R > -20 (overbought) and price < 1d EMA34 (downtrend) with volume spike
# Williams %R identifies reversal points in ranging/bear markets; EMA34 filters for trend alignment to avoid counter-trend trades
# Volume confirmation ensures participation and reduces false signals
# Designed for low trade frequency (~15-25/year on 6h) to minimize fee drag while capturing high-probability reversals
# Works in bull markets via pullback longs in uptrend and in bear markets via rally shorts in downtrend

name = "6h_1dWilliamsR_Extreme_1dEMA34_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Williams %R and EMA calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d Williams %R(14): (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    close_1d = df_1d['close'].values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to 6h timeframe (wait for 1d bar to close)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for dynamic stoploss on 6h
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 34  # warmup for EMA(34)
    
    for i in range(start_idx, n):
        # Volume confirmation: volume > 2.0x 50-period average (moderate to balance frequency and reliability)
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
        curr_ema = ema_34_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and Williams %R extreme
            if volume_spike:
                # Bullish entry: Williams %R oversold (< -80) and price above 1d EMA34 (uptrend)
                if curr_williams_r < -80.0 and curr_close > curr_ema:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: Williams %R overbought (> -20) and price below 1d EMA34 (downtrend)
                elif curr_williams_r > -20.0 and curr_close < curr_ema:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.0 * ATR below entry price OR Williams %R becomes overbought (> -20) signaling reversal
            if curr_close < entry_price - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_williams_r > -20.0:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches 1.5x ATR above entry
            elif curr_close > entry_price + 1.5 * curr_atr:
                signals[i] = 0.0  # full exit
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2.0 * ATR above entry price OR Williams %R becomes oversold (< -80) signaling reversal
            if curr_close > entry_price + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_williams_r < -80.0:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches 1.5x ATR below entry
            elif curr_close < entry_price - 1.5 * curr_atr:
                signals[i] = 0.0  # full exit
            else:
                signals[i] = -0.25
    
    return signals