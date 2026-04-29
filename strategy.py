#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Uses Donchian channel (20-day high/low) for structural breakouts
# 1w EMA50 provides strong multi-week trend filter to avoid counter-trend trades
# Volume spike (2.0x 20-period average) confirms breakout validity with institutional participation
# ATR-based stoploss (2.0x ATR) manages risk while allowing for volatility
# Designed for very low trade frequency (target: 7-25 trades/year) to minimize fee drag
# Works in bull markets via upper band breakouts and in bear markets via lower band breakdowns

name = "1d_Donchian_Breakout_1wEMA50_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate ATR for stoploss (using 14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 20  # warmup for Donchian
    
    for i in range(start_idx, n):
        # Need at least 50 previous days for 1w EMA (but aligned array handles this)
        # Calculate Donchian channels from previous 20 days
        if i < 20:
            signals[i] = 0.0
            continue
            
        # Donchian channels: 20-period high/low
        donch_high = np.max(high[i-20:i])
        donch_low = np.min(low[i-20:i])
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_1w = ema_50_1w_aligned[i]
        curr_atr = atr[i]
        
        # Volume spike confirmation: current volume > 2.0x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        else:
            vol_ma_20 = 0.0
        vol_spike = volume[i] > 2.0 * vol_ma_20 if vol_ma_20 > 0 else False
        
        # Handle exits and stoploss
        if position == 1:  # Long position
            # Stoploss: 2.0 * ATR below entry
            stop_price = entry_price - 2.0 * curr_atr
            # Exit conditions: price below Donchian low OR price below 1w EMA50 OR stoploss hit
            if curr_close < donch_low or curr_close < curr_ema_1w or curr_close < stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Stoploss: 2.0 * ATR above entry
            stop_price = entry_price + 2.0 * curr_atr
            # Exit conditions: price above Donchian high OR price above 1w EMA50 OR stoploss hit
            if curr_close > donch_high or curr_close > curr_ema_1w or curr_close > stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above Donchian high AND price > 1w EMA50 AND volume spike
            if curr_high > donch_high and curr_close > curr_ema_1w and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short entry: price breaks below Donchian low AND price < 1w EMA50 AND volume spike
            elif curr_low < donch_low and curr_close < curr_ema_1w and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals