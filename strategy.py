#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA50 trend + volume confirmation
# Long when price breaks above 20-day high AND price > 1w EMA50 AND volume > 1.5x 20-day avg
# Short when price breaks below 20-day low AND price < 1w EMA50 AND volume > 1.5x 20-day avg
# Exit via ATR-based trailing stop: signal → 0 when price moves against position by 2.5x ATR(14)
# Uses discrete position sizing (0.30) to reduce fee drag. Target: 7-25 trades/year on 1d timeframe.
# Donchian channels provide structural breakout levels, 1w EMA50 filters counter-trend moves in bear markets,
# volume confirmation ensures breakout validity. This combination has worked well on BTC/ETH historically.

name = "1d_Donchian20_1wEMA50_VolumeSpike_ATRStop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1w data
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align EMA50 to 1d timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate ATR(14) for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr2.iloc[0] = tr2.iloc[1] if len(tr2) > 1 else 0
    tr3.iloc[0] = tr3.iloc[1] if len(tr3) > 1 else 0
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: >1.5x 20-day average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = max(20, 50)  # Donchian and EMA50 warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(volume_ma_20[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_ema50 = ema_50_1w_aligned[i]
        curr_dch_high = donchian_high[i]
        curr_dch_low = donchian_low[i]
        curr_close = close[i]
        curr_atr = atr[i]
        
        # Handle position exits and management
        if position == 1:  # Long position
            # Update highest price since entry
            if curr_close > highest_since_entry:
                highest_since_entry = curr_close
            # ATR-based trailing stop: exit if price drops 2.5*ATR from high
            if curr_close < highest_since_entry - 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Update lowest price since entry
            if curr_close < lowest_since_entry:
                lowest_since_entry = curr_close
            # ATR-based trailing stop: exit if price rises 2.5*ATR from low
            if curr_close > lowest_since_entry + 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
                
        else:  # Flat - look for new entries
            # Long when price breaks above Donchian high AND price > 1w EMA50 AND volume confirmation
            if curr_close > curr_dch_high and curr_close > curr_ema50 and vol_conf:
                signals[i] = 0.30
                position = 1
                entry_price = curr_close
                highest_since_entry = curr_close
                lowest_since_entry = curr_close
            # Short when price breaks below Donchian low AND price < 1w EMA50 AND volume confirmation
            elif curr_close < curr_dch_low and curr_close < curr_ema50 and vol_conf:
                signals[i] = -0.30
                position = -1
                entry_price = curr_close
                highest_since_entry = curr_close
                lowest_since_entry = curr_close
            else:
                signals[i] = 0.0
    
    return signals