#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1d EMA50 trend filter and volume confirmation
# Uses daily EMA50 for trend filter to reduce whipsaw in ranging markets
# Donchian(20) breakout from previous 12h range acts as entry signal
# Volume spike confirms breakout validity with 2.0x 20-period average
# ATR-based stoploss (2x ATR) manages risk
# Designed for 12h timeframe with target: 50-150 total trades over 4 years (12-37/year)
# Works in bull markets via trend-following breaks and in bear markets via avoidance of counter-trend trades

name = "12h_Donchian_Breakout_1dEMA50_VolumeSpike_ATRStop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
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
        # Need at least 1 previous bar to calculate Donchian levels
        if i < 1:
            signals[i] = 0.0
            continue
            
        # Calculate Donchian levels from previous 20 bars (12h timeframe)
        # This gives us the 20-period high/low for breakout detection
        if i < 20:
            signals[i] = 0.0
            continue
            
        donchian_high = np.max(high[i-20:i])
        donchian_low = np.min(low[i-20:i])
        
        curr_close = close[i]
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
            # Stoploss: 2 * ATR below entry
            stop_price = entry_price - 2.0 * curr_atr
            # Exit conditions: price below Donchian low OR price below 1d EMA50 OR stoploss hit
            if curr_close < donchian_low or curr_close < curr_ema_1d or curr_close < stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Stoploss: 2 * ATR above entry
            stop_price = entry_price + 2.0 * curr_atr
            # Exit conditions: price above Donchian high OR price above 1d EMA50 OR stoploss hit
            if curr_close > donchian_high or curr_close > curr_ema_1d or curr_close > stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above Donchian high AND price > 1d EMA50 AND volume spike
            if curr_close > donchian_high and curr_close > curr_ema_1d and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short entry: price breaks below Donchian low AND price < 1d EMA50 AND volume spike
            elif curr_close < donchian_low and curr_close < curr_ema_1d and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals