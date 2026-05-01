#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume spike confirmation.
# Long when price breaks above 20-period Donchian high with volume > 2.0x 20-period volume average and price > 1d EMA34.
# Short when price breaks below 20-period Donchian low with volume confirmation and price < 1d EMA34.
# Uses discrete sizing 0.25. ATR(14) stoploss: signal→0 when price moves against position by 2.0*ATR.
# Donchian channels calculated from prior completed 4h bar to avoid look-ahead.
# Volume spike filters low-momentum breakouts. 1d EMA34 ensures trades only in established trends.
# Works in bull (breakouts with strong uptrend) and bear (breakouts with strong downtrend) regimes.
# Target: 20-30 trades/year on 4h timeframe.

name = "4h_Donchian_20_Breakout_1dEMA34_VolumeSpike_v1"
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
    
    # Load 1d data ONCE before loop for EMA and volume filters (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for ATR, EMA, and Donchian
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume spike: current volume > 2.0x 1d volume average
        if vol_ma_1d_aligned[i] <= 0 or np.isnan(vol_ma_1d_aligned[i]):
            volume_spike = False
        else:
            volume_spike = curr_volume > (vol_ma_1d_aligned[i] * 2.0)
        
        # Trend filter: price vs 1d EMA34
        uptrend = curr_close > ema_34_1d_aligned[i]
        downtrend = curr_close < ema_34_1d_aligned[i]
        
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
                 (curr_close < ema_34_1d_aligned[i]):  # trend reversal
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
                 (curr_close > ema_34_1d_aligned[i]):  # trend reversal
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals