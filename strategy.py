#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation
# Uses 1d HTF for Donchian channel calculation (upper/lower 20-period) for strong breakout signals and 1w EMA34 for trend to filter false breakouts.
# Long when price breaks above 1d Donchian upper in uptrend (4h close > 1w EMA34) with volume spike (>2.0x average).
# Short when price breaks below 1d Donchian lower in downtrend (4h close < 1w EMA34) with volume spike.
# Designed for low trade frequency (~19-50/year on 4h) to minimize fee drag while capturing strong directional moves.
# Uses moderate volume confirmation (>2.0x average) and proven Donchian structure to balance signal quality and frequency.
# Stoploss at 2.0 * ATR and take profit at 3.0 * ATR for asymmetric risk-reward (favors winners).
# Works in bull markets via breakout continuation and in bear markets via fade of false breakouts at 1d Donchian levels.
# Focus on BTC/ETH as primary targets.

name = "4h_1dDonchian20_Breakout_1wEMA34_VolumeSpike_v1"
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
    
    # Load 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1d Donchian(20) levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian upper: max(high, 20), lower: min(low, 20)
    donchian_upper = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align 1d Donchian levels to 4h timeframe (wait for 1d bar to close)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    
    # Calculate 1w EMA(34) for trend filter
    close_1w = df_1w['close'].values
    ema_34 = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34)
    
    # Calculate ATR(14) for dynamic stoploss on 4h
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 40  # warmup for EMA(34) and Donchian(20)
    
    for i in range(start_idx, n):
        # Volume confirmation: volume > 2.0x 40-period average
        if i >= 40:
            vol_ma_40 = np.mean(volume[i-40:i])
        elif i > 0:
            vol_ma_40 = np.mean(volume[:i])
        else:
            vol_ma_40 = 0
        volume_spike = volume[i] > (2.0 * vol_ma_40) if i > 0 else False
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_atr = atr[i]
        curr_upper = donchian_upper_aligned[i]
        curr_lower = donchian_lower_aligned[i]
        curr_ema = ema_34_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and trend alignment
            if volume_spike:
                # Bullish entry: price breaks above 1d Donchian upper with 1w uptrend (close > EMA34)
                if curr_close > curr_upper and curr_close > curr_ema:
                    signals[i] = 0.30
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below 1d Donchian lower with 1w downtrend (close < EMA34)
                elif curr_close < curr_lower and curr_close < curr_ema:
                    signals[i] = -0.30
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
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Stoploss: 2.0 * ATR above entry price
            if curr_close > entry_price + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches 3.0x ATR below entry
            elif curr_close < entry_price - 3.0 * curr_atr:
                signals[i] = 0.0  # full exit
            else:
                signals[i] = -0.30
    
    return signals