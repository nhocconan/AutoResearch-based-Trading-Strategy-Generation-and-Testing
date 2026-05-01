#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d Supertrend filter and volume confirmation spike.
# Long when price breaks above Donchian upper band AND Supertrend=1 AND volume > 2.0x 4h volume median.
# Short when price breaks below Donchian lower band AND Supertrend=-1 AND volume > 2.0x 4h volume median.
# Uses discrete sizing 0.25. ATR(14) stoploss: signal→0 when price moves against position by 2.5*ATR.
# Donchian provides clear structure; 1d Supertrend filters longer-term trend (works in bull/bear).
# Volume confirmation ensures momentum. Target: 20-35 trades/year on 4h timeframe.

name = "4h_Donchian20_Supertrend1d_Volume_v1"
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
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 4h volume median (20-period for stability)
    vol_median_4h = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # Calculate 1d Supertrend (more stable than EMA)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Supertrend parameters
    atr_period = 10
    multiplier = 3.0
    
    # Calculate ATR for Supertrend
    hl2 = (df_1d['high'] + df_1d['low']) / 2
    tr1_1d = df_1d['high'].diff()
    tr2_1d = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3_1d = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr_1d = pd.concat([tr1_1d, tr2_1d, tr3_1d], axis=1).max(axis=1)
    tr_1d.iloc[0] = np.max([df_1d['high'].iloc[0] - df_1d['low'].iloc[0], 
                           np.abs(df_1d['high'].iloc[0] - df_1d['close'].iloc[0]), 
                           np.abs(df_1d['low'].iloc[0] - df_1d['close'].iloc[0])])
    atr_1d = tr_1d.rolling(window=atr_period, min_periods=atr_period).mean()
    
    # Calculate upper and lower bands
    upper_band = hl2 + (multiplier * atr_1d)
    lower_band = hl2 - (multiplier * atr_1d)
    
    # Initialize Supertrend
    supertrend = pd.Series(index=df_1d.index, dtype=float)
    direction = pd.Series(index=df_1d.index, dtype=int)
    
    for i in range(len(df_1d)):
        if i == 0:
            supertrend.iloc[i] = upper_band.iloc[i]
            direction.iloc[i] = 1
        else:
            if close_1d := df_1d['close'].iloc[i]:
                if supertrend.iloc[i-1] == upper_band.iloc[i-1]:
                    if close_1d <= upper_band.iloc[i]:
                        supertrend.iloc[i] = upper_band.iloc[i]
                        direction.iloc[i] = -1
                    else:
                        supertrend.iloc[i] = lower_band.iloc[i]
                        direction.iloc[i] = 1
                else:
                    if close_1d >= lower_band.iloc[i]:
                        supertrend.iloc[i] = lower_band.iloc[i]
                        direction.iloc[i] = 1
                    else:
                        supertrend.iloc[i] = upper_band.iloc[i]
                        direction.iloc[i] = -1
            else:
                supertrend.iloc[i] = supertrend.iloc[i-1]
                direction.iloc[i] = direction.iloc[i-1]
    
    supertrend_values = supertrend.values
    direction_values = direction.values
    
    # Align Supertrend and direction to 4h timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_1d, supertrend_values)
    direction_aligned = align_htf_to_ltf(prices, df_1d, direction_values)
    
    # Calculate Donchian channels (20-period) on 4h data
    donchian_window = 20
    upper_channel = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lower_channel = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for ATR, Supertrend, Donchian, and volume
    start_idx = max(100, donchian_window)
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or 
            np.isnan(supertrend_aligned[i]) or 
            np.isnan(direction_aligned[i]) or 
            np.isnan(upper_channel[i]) or 
            np.isnan(lower_channel[i]) or 
            np.isnan(vol_median_4h[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Trend filter: Supertrend direction
        uptrend = direction_aligned[i] == 1
        downtrend = direction_aligned[i] == -1
        
        # Volume confirmation: current volume > 2.0x 4h volume median (stricter to reduce trades)
        if vol_median_4h[i] <= 0 or np.isnan(vol_median_4h[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_4h[i] * 2.0)
        
        if position == 0:  # Flat - look for new entries
            # Long: price > upper channel AND uptrend AND volume spike
            if curr_close > upper_channel[i] and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: price < lower channel AND downtrend AND volume spike
            elif curr_close < lower_channel[i] and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Stoploss: price moves against position by 2.5*ATR
            if curr_close < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks below lower channel OR trend turns down
            elif curr_close < lower_channel[i] or not uptrend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: price moves against position by 2.5*ATR
            if curr_close > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price breaks above upper channel OR trend turns up
            elif curr_close > upper_channel[i] or not downtrend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals