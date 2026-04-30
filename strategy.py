#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using weekly Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation
# Uses 1w HTF for Donchian breakout levels to capture major structural breaks and 1d HTF for EMA34 trend filter.
# Long when price breaks above weekly Donchian high in uptrend (1d close > 1d EMA34) with volume spike (>2.0x average).
# Short when price breaks below weekly Donchian low in downtrend (1d close < 1d EMA34) with volume spike.
# Designed for very low trade frequency (~12-25/year on 12h) to minimize fee drag while capturing strong directional moves.
# Uses volume confirmation with moderate threshold (>2.0x average) to balance signal quality and trade frequency.
# Stoploss at 2.5 * ATR and no fixed take profit - lets winners run with trailing structure via Donchian levels.
# Works in bull markets via breakout continuation and in bear markets via fade of false breakouts at Donchian levels.
# Focus on BTC/ETH as primary targets with SOL as secondary.

name = "12h_1wDonchian20_Breakout_1dEMA34_VolumeSpike_v1"
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
    
    # Load 1w data ONCE before loop for Donchian calculations
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1w Donchian levels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Donchian: upper = max(high, 20), lower = min(low, 20)
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align 1w Donchian levels to 12h timeframe (wait for 1w bar to close)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA to 12h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for dynamic stoploss on 12h
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 50  # warmup for EMA(34) and Donchian(20)
    
    for i in range(start_idx, n):
        # Volume confirmation: volume > 2.0x 50-period average (moderate to balance frequency)
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
        curr_dch_high = donchian_high_aligned[i]
        curr_dch_low = donchian_low_aligned[i]
        curr_ema = ema_34_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and trend alignment
            if volume_spike:
                # Bullish entry: price breaks above 1w Donchian high with 1d uptrend (close > EMA34)
                if curr_close > curr_dch_high and curr_close > curr_ema:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below 1w Donchian low with 1d downtrend (close < EMA34)
                elif curr_close < curr_dch_low and curr_close < curr_ema:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.5 * ATR below entry price OR price breaks 1w Donchian low (structure break)
            if curr_close < entry_price - 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close < curr_dch_low:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price drops below 1w Donchian low (structure break)
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2.5 * ATR above entry price OR price breaks 1w Donchian high (structure break)
            if curr_close > entry_price + 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close > curr_dch_high:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price rises above 1w Donchian high (structure break)
            else:
                signals[i] = -0.25
    
    return signals