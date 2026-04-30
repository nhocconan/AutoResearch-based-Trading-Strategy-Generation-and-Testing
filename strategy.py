#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Williams %R extreme readings with 12h EMA34 trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions; extreme readings (< -80 or > -20) combined with 12h trend direction
# provide high-probability reversal entries in ranging/choppy markets. Volume confirmation (>2.0x average) filters weak signals.
# Designed for low trade frequency (~15-25/year on 4h) to minimize fee drag while capturing mean-reversion moves.
# Works in bull markets via fade of overextended rallies and in bear markets via fade of panic selling.
# Uses ATR-based dynamic stoploss (1.5x) and take profit (1.0x) for asymmetric risk/reward.
# Focus on BTC/ETH as primary targets with proven edge in ranging regimes.

name = "4h_1dWilliamsR_Extreme_12hEMA34_VolumeConfirm_v1"
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
    
    # Load 12h data ONCE before loop for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 14-period Williams %R on 1d: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Rolling highest high and lowest low for 14-period
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Williams %R: values between -100 and 0
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align 1d Williams %R to 4h timeframe (wait for 1d bar to close)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate 12h EMA(34) for trend filter
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 12h EMA to 4h timeframe
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate ATR(10) for dynamic risk management on 4h
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 50  # warmup for EMA(34) and ATR(10)
    
    for i in range(start_idx, n):
        # Volume confirmation: volume > 2.0x 20-period average (moderate threshold)
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        elif i > 0:
            vol_ma_20 = np.mean(volume[:i])
        else:
            vol_ma_20 = 0
        volume_spike = volume[i] > (2.0 * vol_ma_20) if i > 0 else False
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_atr = atr[i]
        curr_williams_r = williams_r_aligned[i]
        curr_ema = ema_34_12h_aligned[i]
        
        if position == 0:  # Flat - look for mean-reversion entries
            # Require volume confirmation and extreme Williams %R
            if volume_spike:
                # Bullish entry: Williams %R deeply oversold (< -80) with 12h uptrend (close > EMA34)
                if curr_williams_r < -80 and curr_close > curr_ema:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: Williams %R deeply overbought (> -20) with 12h downtrend (close < EMA34)
                elif curr_williams_r > -20 and curr_close < curr_ema:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 1.5 * ATR below entry price
            if curr_close < entry_price - 1.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            # Take profit: 1.0 * ATR above entry (asymmetric R/R)
            elif curr_close > entry_price + 1.0 * curr_atr:
                signals[i] = 0.0  # full exit
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 1.5 * ATR above entry price
            if curr_close > entry_price + 1.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            # Take profit: 1.0 * ATR below entry
            elif curr_close < entry_price - 1.0 * curr_atr:
                signals[i] = 0.0  # full exit
            else:
                signals[i] = -0.25
    
    return signals