#!/usr/bin/env python3
"""
6h_1w_Liquidity_Trap_V1
Hypothesis: Combines weekly liquidity traps (price rejection at weekly high/low) with 60-period EMA trend filter on 6h timeframe.
Enters long when price rejects weekly low with bullish engulfing candle and price above EMA60.
Enters short when price rejects weekly high with bearish engulfing candle and price below EMA60.
Designed for 15-30 trades/year per symbol with high win rate by avoiding false breakouts.
Works in bull/bear by following 60-period EMA trend direction - avoids counter-trend whipsaws.
"""

from typing import Tuple
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_Liquidity_Trap_V1"
timeframe = "6h"
leverage = 1.0

def engulfing_candle(open_price, close_price, prev_open, prev_close) -> Tuple[bool, bool]:
    """Returns (bullish_engulfing, bearish_engulfing)"""
    bullish = close_price > open_price and prev_close < prev_open and \
              close_price >= prev_open and open_price <= prev_close
    bearish = close_price < open_price and prev_close > prev_open and \
              close_price <= prev_open and open_price >= prev_close
    return bullish, bearish

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    open_price = prices['open'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for liquidity levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 60-period EMA for trend filter
    ema_60 = pd.Series(close).ewm(span=60, adjust=False, min_periods=60).mean().values
    
    # Weekly high and low from previous week
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    weekly_high = np.full_like(high_1w, np.nan)
    weekly_low = np.full_like(low_1w, np.nan)
    
    for i in range(1, len(high_1w)):
        weekly_high[i] = high_1w[i-1]
        weekly_low[i] = low_1w[i-1]
    
    # Align weekly levels to 6h timeframe
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if any required data is invalid
        if (np.isnan(weekly_high_aligned[i]) or np.isnan(weekly_low_aligned[i]) or 
            np.isnan(ema_60[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Engulfing candle detection (need previous candle)
        bullish_eng, bearish_eng = engulfing_candle(
            open_price[i], close[i], open_price[i-1], close[i-1]
        )
        
        # Liquidity trap conditions
        # Long: bullish engulfing near weekly low + above EMA60
        near_weekly_low = low[i] <= weekly_low_aligned[i] * 1.002  # within 0.2% of weekly low
        long_trap = bullish_eng and near_weekly_low and close[i] > ema_60[i]
        
        # Short: bearish engulfing near weekly high + below EMA60
        near_weekly_high = high[i] >= weekly_high_aligned[i] * 0.998  # within 0.2% of weekly high
        short_trap = bearish_eng and near_weekly_high and close[i] < ema_60[i]
        
        # Exit conditions: opposite engulfing candle or trend violation
        if position == 1:
            exit_condition = bearish_eng or close[i] < ema_60[i]
        elif position == -1:
            exit_condition = bullish_eng or close[i] > ema_60[i]
        else:
            exit_condition = False
        
        # Priority: entry > exit > hold
        if long_trap and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_trap and position != -1:
            position = -1
            signals[i] = -0.25
        elif position != 0 and exit_condition:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals