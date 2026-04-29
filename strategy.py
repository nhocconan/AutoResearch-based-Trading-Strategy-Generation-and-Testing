#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and ATR-based volatility filter
# Works in bull markets via breakout continuation and in bear markets via trend filter avoidance of counter-trend trades
# ATR filter ensures trades only occur during sufficient volatility, reducing whipsaw in low-volatility ranging markets
# Uses discrete position sizing (0.25) to minimize fee churn and manage drawdown
# Target trade frequency: 20-50 per year to avoid fee drag while maintaining statistical significance

name = "4h_Donchian_Breakout_1dEMA34_ATRFilter_v1"
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
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for volatility filter and stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 34  # warmup for EMA and Donchian
    
    for i in range(start_idx, n):
        # Need at least 1 previous bar for Donchian calculation
        if i < 1:
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_1d = ema_34_1d_aligned[i]
        curr_atr = atr[i]
        prev_highest = highest_high[i-1]
        prev_lowest = lowest_low[i-1]
        
        # Volatility filter: ATR > 0.5 * 20-period average ATR (avoid low-volatility whipsaw)
        if i >= 20:
            atr_ma_20 = np.mean(atr[i-20:i])
        else:
            atr_ma_20 = 0.0
        vol_filter = curr_atr > 0.5 * atr_ma_20 if atr_ma_20 > 0 else True
        
        # Handle exits and stoploss
        if position == 1:  # Long position
            # Stoploss: 2 * ATR below entry
            stop_price = entry_price - 2.0 * curr_atr
            # Exit conditions: price below Donchian lower band OR price below 1d EMA34 OR stoploss hit
            if curr_close < prev_lowest or curr_close < curr_ema_1d or curr_close < stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Stoploss: 2 * ATR above entry
            stop_price = entry_price + 2.0 * curr_atr
            # Exit conditions: price above Donchian upper band OR price above 1d EMA34 OR stoploss hit
            if curr_close > prev_highest or curr_close > curr_ema_1d or curr_close > stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above Donchian upper band AND price > 1d EMA34 AND volatility filter
            if curr_high > prev_highest and curr_close > curr_ema_1d and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short entry: price breaks below Donchian lower band AND price < 1d EMA34 AND volatility filter
            elif curr_low < prev_lowest and curr_close < curr_ema_1d and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals