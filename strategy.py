#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation
# Donchian channels provide clear breakout levels; EMA34 filters for higher timeframe trend alignment.
# Volume confirmation ensures breakouts have institutional participation. Designed to work in both
# bull and bear markets by requiring volume and trend alignment to avoid false breakouts.
# Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe.

name = "12h_Donchian20_Breakout_1dEMA34_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Donchian and EMA calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # need enough for EMA(34)
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (20-period) using prior 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian upper/lower based on prior 20 1d bars (exclude current)
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align Donchian levels to 12h timeframe (wait for completed 1d bar)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().shift(1).values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate ATR(14) for dynamic stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Volume confirmation: volume > 2.0x 30-period average
        vol_ma_30 = np.mean(volume[max(0, i-30):i])
        volume_spike = volume[i] > (2.0 * vol_ma_30)
        
        curr_close = close[i]
        curr_donchian_high = donchian_high_aligned[i]
        curr_donchian_low = donchian_low_aligned[i]
        curr_ema = ema_34_aligned[i]
        curr_atr = atr[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and trend alignment
            if volume_spike:
                # Bullish entry: price breaks above Donchian high with 1d uptrend
                if curr_close > curr_donchian_high and curr_close > curr_ema:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below Donchian low with 1d downtrend
                elif curr_close < curr_donchian_low and curr_close < curr_ema:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.0 * ATR below entry price OR price breaks Donchian low
            if curr_close < entry_price - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close < curr_donchian_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2.0 * ATR above entry price OR price breaks Donchian high
            if curr_close > entry_price + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif curr_close > curr_donchian_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals