#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter, volume confirmation (>1.5x 20-bar MA), and ATR-based stoploss.
# Long when price breaks above Donchian upper channel, price > 1d EMA50, and volume spike.
# Short when price breaks below Donchian lower channel, price < 1d EMA50, and volume spike.
# Uses discrete sizing (0.25) to minimize fee churn and ATR stoploss for risk management.
# Target: 75-200 total trades over 4 years (19-50/year) to stay within fee drag limits.
# Works in both bull and bear markets via trend filter and volatility-based breakouts.

name = "4h_Donchian20_Breakout_1dEMA50_Trend_VolumeSpike_ATRStop_v1"
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
    
    # 1d HTF data for EMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) on 1d close
    ema_1d_50 = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA to 4h timeframe
    ema_1d_50_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_50)
    
    # Donchian Channel (20-period) on 4h data
    high_ma_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ATR(14) for stoploss and volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR is just high-low
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # Track entry price for ATR stoploss
    
    # Start after warmup for all indicators
    start_idx = 50  # Need 50 for 1d EMA50 and Donchian(20)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_1d_50_aligned[i]) or np.isnan(high_ma_20[i]) or 
            np.isnan(low_ma_20[i]) or np.isnan(atr_14[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        # Donchian breakout conditions
        breakout_up = curr_high > high_ma_20[i-1]  # Break above upper channel
        breakout_down = curr_low < low_ma_20[i-1]  # Break below lower channel
        
        if position == 0:  # Flat - look for new entries
            # Long: Break above Donchian upper, price > 1d EMA50, volume spike
            if breakout_up and curr_close > ema_1d_50_aligned[i] and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Break below Donchian lower, price < 1d EMA50, volume spike
            elif breakout_down and curr_close < ema_1d_50_aligned[i] and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # ATR-based stoploss: exit if price drops below entry - 2.0 * ATR
            stop_loss = entry_price - 2.0 * atr_14[i]
            # Exit on stoploss, price below 1d EMA50, or break below Donchian lower
            if curr_close < stop_loss or curr_close < ema_1d_50_aligned[i] or curr_low < low_ma_20[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # ATR-based stoploss: exit if price rises above entry + 2.0 * ATR
            stop_loss = entry_price + 2.0 * atr_14[i]
            # Exit on stoploss, price above 1d EMA50, or break above Donchian upper
            if curr_close > stop_loss or curr_close > ema_1d_50_aligned[i] or curr_high > high_ma_20[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals