#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# Donchian breakouts capture strong momentum moves in both bull and bear markets
# 1d EMA50 provides robust trend filter to avoid counter-trend trades
# Volume spike (2.0x 20-period average) confirms breakout validity with institutional participation
# ATR-based trailing stop (2.5x ATR) allows trends to run while managing risk
# Target trade frequency: 20-50 trades/year to minimize fee drag while capturing major moves
# Works in bull markets via upper band breakouts and bear markets via lower band breakdowns

name = "4h_Donchian_Breakout_1dEMA50_VolumeConfirm_v3"
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
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
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
    
    start_idx = 50  # warmup for EMA
    
    for i in range(start_idx, n):
        # Need at least 20 previous 4h bars for Donchian calculation
        if i < 20:
            signals[i] = 0.0
            continue
            
        # Calculate Donchian channels from previous 20 periods
        lookback_start = i - 20
        lookback_end = i - 1  # exclude current bar
        
        upper_channel = np.max(high[lookback_start:lookback_end+1])
        lower_channel = np.min(low[lookback_start:lookback_end+1])
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_1d = ema_50_1d_aligned[i]
        curr_atr = atr[i]
        
        # Volume spike confirmation: current volume > 2.0x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        else:
            vol_ma_20 = 0.0
        vol_spike = volume[i] > 2.0 * vol_ma_20 if vol_ma_20 > 0 else False
        
        # Handle exits and stoploss
        if position == 1:  # Long position
            # Trailing stop: 2.5 * ATR below highest high since entry
            # Simplified: use 2.5 * ATR below current close as proxy
            stop_price = curr_close - 2.5 * curr_atr
            # Exit conditions: price below lower channel OR trailing stop hit
            if curr_close < lower_channel or curr_close < stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Trailing stop: 2.5 * ATR above lowest low since entry
            # Simplified: use 2.5 * ATR above current close as proxy
            stop_price = curr_close + 2.5 * curr_atr
            # Exit conditions: price above upper channel OR trailing stop hit
            if curr_close > upper_channel or curr_close > stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above upper channel AND price > 1d EMA50 AND volume spike
            if curr_high > upper_channel and curr_close > curr_ema_1d and vol_spike:
                signals[i] = 0.30
                position = 1
                entry_price = curr_close
            # Short entry: price breaks below lower channel AND price < 1d EMA50 AND volume spike
            elif curr_low < lower_channel and curr_close < curr_ema_1d and vol_spike:
                signals[i] = -0.30
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals