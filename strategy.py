#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume spike confirmation.
# Long when price breaks above 20-period Donchian high with volume > 2.0x 12h volume average and price > 12h EMA50.
# Short when price breaks below 20-period Donchian low with volume confirmation and price < 12h EMA50.
# Uses discrete sizing 0.25. ATR(14) stoploss: signal→0 when price moves against position by 2.0*ATR.
# Donchian channels calculated from prior completed 4h bar to avoid look-ahead.
# Volume spike filters low-momentum breakouts. 12h EMA50 ensures trades only in established trends.
# Works in bull (breakouts with strong uptrend) and bear (breakouts with strong downtrend) regimes.
# Target: 25-35 trades/year on 4h timeframe.

name = "4h_Donchian_20_Breakout_12hEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
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
    
    # Load 12h data ONCE before loop for EMA and volume filters (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 12h volume average (20-period)
    vol_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for ATR, EMA, and Donchian
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(vol_ma_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume spike: current volume > 2.0x 12h volume average
        if vol_ma_12h_aligned[i] <= 0 or np.isnan(vol_ma_12h_aligned[i]):
            volume_spike = False
        else:
            volume_spike = curr_volume > (vol_ma_12h_aligned[i] * 2.0)
        
        # Trend filter: price vs 12h EMA50
        uptrend = curr_close > ema_50_12h_aligned[i]
        downtrend = curr_close < ema_50_12h_aligned[i]
        
        # Load 4h data ONCE before loop for Donchian channels
        df_4h = get_htf_data(prices, '4h')
        if len(df_4h) < 20:
            signals[i] = 0.0
            continue
        
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        
        # Calculate Donchian channels for each 4h bar (using previous completed bar)
        # Upper channel = max(high of last 20 bars)
        # Lower channel = min(low of last 20 bars)
        # Use previous completed bars to avoid look-ahead
        if i < 20:  # Need at least 20 previous 4h bars
            signals[i] = 0.0
            continue
            
        # Get the index range for the last 20 completed 4h bars
        start_4h = i - 20
        end_4h = i - 1
        
        if start_4h < 0:
            signals[i] = 0.0
            continue
            
        # Calculate Donchian levels using previous 20 completed bars
        high_slice = high_4h[start_4h:end_4h+1]
        low_slice = low_4h[start_4h:end_4h+1]
        
        if len(high_slice) < 20:
            signals[i] = 0.0
            continue
            
        upper_channel = np.max(high_slice)
        lower_channel = np.min(low_slice)
        
        if position == 0:  # Flat - look for new entries
            # Long: Donchian breakout up AND volume spike AND uptrend
            if (curr_high > upper_channel and 
                volume_spike and 
                uptrend):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Donchian breakout down AND volume spike AND downtrend
            elif (curr_low < lower_channel and 
                  volume_spike and 
                  downtrend):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Stoploss: price moves against position by 2.0*ATR
            if curr_close < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price re-enters Donchian channels OR trend reverses
            elif (curr_low >= lower_channel and curr_low <= upper_channel) or \
                 (curr_close < ema_50_12h_aligned[i]):  # trend reversal
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: price moves against position by 2.0*ATR
            if curr_close > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price re-enters Donchian channels OR trend reverses
            elif (curr_high >= lower_channel and curr_high <= upper_channel) or \
                 (curr_close > ema_50_12h_aligned[i]):  # trend reversal
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals