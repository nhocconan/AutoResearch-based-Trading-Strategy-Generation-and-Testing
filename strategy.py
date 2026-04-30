#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly Williams %R extremes with 1d EMA34 trend filter and volume confirmation
# Williams %R > -20 = overbought (short signal), < -80 = oversold (long signal) on weekly timeframe
# Only take signals when aligned with 1d EMA34 trend to avoid counter-trend whipsaws
# Volume spike confirmation ensures institutional participation
# Designed for low trade frequency (~15-30/year on 6h) to minimize fee drag while capturing reversal extremes
# Works in bull markets via buying oversold dips and in bear markets via selling overbought rallies
# Focus on BTC/ETH as primary targets (avoid SOL-only bias)

name = "6h_1wWilliamsR_Extreme_1dEMA34_VolumeSpike_v1"
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
    
    # Load weekly data ONCE before loop for Williams %R calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for EMA and volume calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate weekly Williams %R(14)
    highest_high_1w = pd.Series(df_1w['high']).rolling(window=14, min_periods=14).max().values
    lowest_low_1w = pd.Series(df_1w['low']).rolling(window=14, min_periods=14).min().values
    close_1w = df_1w['close'].values
    williams_r = -100 * (highest_high_1w - close_1w) / (highest_high_1w - lowest_low_1w)
    
    # Align weekly Williams %R to 6h timeframe (wait for weekly bar to close)
    williams_r_aligned = align_htf_to_ltf(prices, df_1w, williams_r)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA to 6h timeframe
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
    
    start_idx = 100  # warmup for EMA(34) and Williams %R
    
    for i in range(start_idx, n):
        # Volume confirmation: volume > 2.0x 50-period average (strict to reduce trades)
        if i >= 50:
            vol_ma_50 = np.mean(volume[i-50:i])
        elif i > 0:
            vol_ma_50 = np.mean(volume[:i])
        else:
            vol_ma_50 = 0
        volume_spike = volume[i] > (2.0 * vol_ma_50) if i > 0 else False
        
        curr_close = close[i]
        curr_atr = atr[i]
        curr_williams = williams_r_aligned[i]
        curr_ema = ema_34_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and trend alignment
            if volume_spike:
                # Bullish entry: weekly Williams %R < -80 (oversold) with 1d uptrend (close > EMA34)
                if curr_williams < -80 and curr_close > curr_ema:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: weekly Williams %R > -20 (overbought) with 1d downtrend (close < EMA34)
                elif curr_williams > -20 and curr_close < curr_ema:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.0 * ATR below entry price OR weekly Williams %R > -20 (overbought reversal)
            if curr_close < entry_price - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_williams > -20:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches 1.5x ATR above entry OR weekly Williams %R > -50 (mean reversion)
            elif curr_close > entry_price + 1.5 * curr_atr:
                signals[i] = 0.10  # reduce position
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2.0 * ATR above entry price OR weekly Williams %R < -80 (oversold reversal)
            if curr_close > entry_price + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_williams < -80:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches 1.5x ATR below entry OR weekly Williams %R < -50 (mean reversion)
            elif curr_close < entry_price - 1.5 * curr_atr:
                signals[i] = -0.10  # reduce position
            else:
                signals[i] = -0.25
    
    return signals