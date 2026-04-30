#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
# Long when price breaks above 20-bar high, price > 1d EMA50, and volume > 1.8x 20-bar avg.
# Short when price breaks below 20-bar low, price < 1d EMA50, and volume > 1.8x 20-bar avg.
# Exit via ATR-based trailing stop: long exits when price drops 2.5x ATR from peak, 
# short exits when price rises 2.5x ATR from trough.
# Uses 1d EMA50 for higher timeframe trend alignment, targeting 20-50 trades/year on 4h.
# Trend filter avoids counter-trend trades, volume confirmation reduces false signals.
# ATR stoploss manages risk without look-ahead. Works in bull markets via breakouts and 
# in bear markets via short breakdowns with trend alignment. Focus on BTC/ETH.

name = "4h_Donchian20_1dEMA50_VolumeConfirm_ATRStop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Donchian channels (20-bar)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    long_peak = 0.0
    short_trough = 0.0
    
    start_idx = 60  # warmup for EMA50, ATR, and Donchian
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(atr[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        curr_highest_high = highest_high[i]
        curr_lowest_low = lowest_low[i]
        curr_atr = atr[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above 20-bar high, price > 1d EMA50, volume spike
            if (curr_close > curr_highest_high and 
                curr_close > curr_ema_50_1d and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
                long_peak = curr_close
            # Short: price breaks below 20-bar low, price < 1d EMA50, volume spike
            elif (curr_close < curr_lowest_low and 
                  curr_close < curr_ema_50_1d and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
                short_trough = curr_close
        
        elif position == 1:  # Long position
            # Update peak
            if curr_close > long_peak:
                long_peak = curr_close
            # ATR trailing stop: exit when price drops 2.5x ATR from peak
            if curr_close <= long_peak - 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Update trough
            if curr_close < short_trough:
                short_trough = curr_close
            # ATR trailing stop: exit when price rises 2.5x ATR from trough
            if curr_close >= short_trough + 2.5 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals