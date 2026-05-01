#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and ATR-based stoploss.
# Uses 1w EMA50 for robust long-term trend alignment that works in both bull and bear markets.
# Long when price breaks above 20-day high AND price > 1w EMA50.
# Short when price breaks below 20-day low AND price < 1w EMA50.
# Exit on ATR(14) trailing stop (3x ATR) or trend reversal.
# Uses discrete sizing 0.25 to manage drawdown. Target: 30-100 total trades over 4 years (7-25/year).

name = "1d_Donchian20_1wEMA50_Trend_ATRStop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 calculation
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # 1d trend: price above/below 1w EMA50
    price_above_ema = close > ema_50_aligned
    price_below_ema = close < ema_50_aligned
    
    # Calculate ATR(14) for 1d
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_high = 0.0
    lowest_low = 0.0
    
    start_idx = 50  # warmup for EMA, ATR, and Donchian
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(atr[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: breakout above Donchian high AND price > 1w EMA50
            if curr_high > donchian_high[i] and price_above_ema[i]:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                highest_high = curr_high
            # Short: breakout below Donchian low AND price < 1w EMA50
            elif curr_low < donchian_low[i] and price_below_ema[i]:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                lowest_low = curr_low
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            highest_high = max(highest_high, curr_high)
            # Exit: ATR trailing stop (3x ATR) OR price < 1w EMA50 (trend change)
            if curr_close < highest_high - 3.0 * atr[i] or not price_above_ema[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            lowest_low = min(lowest_low, curr_low)
            # Exit: ATR trailing stop (3x ATR) OR price > 1w EMA50 (trend change)
            if curr_close > lowest_low + 3.0 * atr[i] or not price_below_ema[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals