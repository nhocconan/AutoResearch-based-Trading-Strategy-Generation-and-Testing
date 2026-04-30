#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation (>2.0x average)
# Uses 4h timeframe for balance of signal quality and trade frequency
# 1d EMA34 provides medium-term trend filter to avoid counter-trend trades
# Volume confirmation >2.0x 20-period average ensures breakout legitimacy
# Discrete position sizing: 0.30 for entries, 0.15 for scaling out
# ATR-based stoploss: exit when price moves 2.5*ATR against position
# Works in bull/bear/sideways: breakouts capture momentum, volume confirms, trend filter avoids fakes

name = "4h_Donchian20_1dEMA34_Volume_ATR_v1"
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
    
    # Calculate ATR for stoploss (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 4h Donchian channels (20-period) from previous bar to avoid look-ahead
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Breakout conditions
    breakout_up = close > donchian_high
    breakout_down = close < donchian_low
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(100, 20, 34)  # warmup for Donchian (20), volume MA (20), EMA (34), ATR (14)
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ma_20[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_breakout_up = breakout_up[i]
        curr_breakout_down = breakout_down[i]
        curr_volume_confirm = volume_confirm[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        curr_atr = atr[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade on breakout with volume confirmation and trend filter
            if curr_volume_confirm:
                # Bullish breakout: price above Donchian high + above 1d EMA34
                if curr_breakout_up and curr_close > curr_ema_34_1d:
                    signals[i] = 0.30
                    position = 1
                    entry_price = curr_close
                # Bearish breakout: price below Donchian low + below 1d EMA34
                elif curr_breakout_down and curr_close < curr_ema_34_1d:
                    signals[i] = -0.30
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit conditions: stoploss or Donchian low break (reversal)
            stop_loss = entry_price - 2.5 * curr_atr
            if curr_close < stop_loss or curr_close < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit conditions: stoploss or Donchian high break (reversal)
            stop_loss = entry_price + 2.5 * curr_atr
            if curr_close > stop_loss or curr_close > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals