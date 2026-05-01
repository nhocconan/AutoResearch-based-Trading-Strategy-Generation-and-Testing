#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation.
# Long when price breaks above Donchian upper band (20-period high) with volume > 1.5x 12h volume average and price > 1d EMA34.
# Short when price breaks below Donchian lower band (20-period low) with volume confirmation and price < 1d EMA34.
# Uses discrete sizing 0.25. ATR(14) stoploss: signal→0 when price moves against position by 2.0*ATR.
# Donchian bands calculated from prior completed 12h bar to avoid look-ahead.
# Volume spike filters low-momentum breakouts. 1d EMA34 ensures trades only in established trends.
# Works in bull (breakouts with strong uptrend) and bear (breakouts with strong downtrend) regimes.
# Target: 12-37 trades/year on 12h timeframe.

name = "12h_Donchian20_Breakout_1dEMA34_Volume_v1"
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
        
        # Volume spike: current volume > 1.5x 1d volume average
        if vol_ma_1d_aligned[i] <= 0 or np.isnan(vol_ma_1d_aligned[i]):
            volume_spike = False
        else:
            volume_spike = curr_volume > (vol_ma_1d_aligned[i] * 1.5)
        
        # Trend filter: price vs 1d EMA34
        uptrend = curr_close > ema_34_1d_aligned[i]
        downtrend = curr_close < ema_34_1d_aligned[i]
        
        # Load 12h data ONCE before loop for Donchian bands
        df_12h = get_htf_data(prices, '12h')
        if len(df_12h) < 20:
            signals[i] = 0.0
            continue
        
        high_12h = df_12h['high'].values
        low_12h = df_12h['low'].values
        close_12h = df_12h['close'].values
        
        # Calculate Donchian bands for each 12h bar (using previous completed bar)
        # Need at least one completed 12h bar
        if i < 1:  # Need at least 1 previous 12h bar
            signals[i] = 0.0
            continue
            
        # Get the previous completed 12h bar
        prev_12h_idx = i - 1
        
        if prev_12h_idx < 0:
            signals[i] = 0.0
            continue
            
        # Calculate Donchian bands using previous completed 12h bar (lookback 20)
        lookback_start = max(0, prev_12h_idx - 19)
        lookback_end = prev_12h_idx + 1
        
        if lookback_end - lookback_start < 20:
            signals[i] = 0.0
            continue
            
        high_window = high_12h[lookback_start:lookback_end]
        low_window = low_12h[lookback_start:lookback_end]
        
        upper_band = np.max(high_window)
        lower_band = np.min(low_window)
        
        if position == 0:  # Flat - look for new entries
            # Long: Donchian upper band breakout up AND volume spike AND uptrend
            if (curr_high > upper_band and 
                volume_spike and 
                uptrend):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Donchian lower band breakout down AND volume spike AND downtrend
            elif (curr_low < lower_band and 
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
            # Exit: price re-enters Donchian bands OR trend reverses
            elif (curr_low >= lower_band and curr_low <= upper_band) or \
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
            # Exit: price re-enters Donchian bands OR trend reverses
            elif (curr_high >= lower_band and curr_high <= upper_band) or \
                 (curr_close > ema_34_1d_aligned[i]):  # trend reversal
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals