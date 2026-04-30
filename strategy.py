#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Donchian channel (20) breakout with 1w EMA200 trend filter and volume confirmation
# Uses 1d HTF for Donchian calculation (key support/resistance structure) and 1w EMA for long-term trend.
# Long when price breaks above 1d Donchian upper in 1w uptrend with volume spike.
# Short when price breaks below 1d Donchian lower in 1w downtrend with volume spike.
# Designed for low trade frequency (~20-30/year on 4h) to minimize fee drag while capturing strong directional moves.
# Focus on BTC/ETH as primary targets.

name = "4h_1dDonchian20_1wEMA200_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d and 1w data ONCE before loop for Donchian and EMA calculations
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1d Donchian channel (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_20_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_20_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align 1d Donchian levels to 4h timeframe (wait for 1d bar to close)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_20_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_20_low)
    
    # Calculate 1w EMA(200) for long-term trend filter
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1w EMA to 4h timeframe
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
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
        # Volume confirmation: volume > 2.0x 50-period average (moderate to balance trades)
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
        curr_donch_high = donchian_high_aligned[i]
        curr_donch_low = donchian_low_aligned[i]
        curr_ema = ema_200_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and trend alignment
            if volume_spike:
                # Bullish entry: price breaks above 1d Donchian high with 1w uptrend (close > EMA200)
                if curr_close > curr_donch_high and curr_close > curr_ema:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below 1d Donchian low with 1w downtrend (close < EMA200)
                elif curr_close < curr_donch_low and curr_close < curr_ema:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.0 * ATR below entry price OR price breaks 1d Donchian low (reversal signal)
            if curr_close < entry_price - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close < curr_donch_low:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches 2.0x ATR above entry OR touches 1d Donchian high (mean reversion)
            elif curr_close > entry_price + 2.0 * curr_atr:
                signals[i] = 0.10  # reduce position
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2.0 * ATR above entry price OR price breaks 1d Donchian high (reversal signal)
            if curr_close > entry_price + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close > curr_donch_high:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches 2.0x ATR below entry OR touches 1d Donchian low (mean reversion)
            elif curr_close < entry_price - 2.0 * curr_atr:
                signals[i] = -0.10  # reduce position
            else:
                signals[i] = -0.25
    
    return signals