#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and ATR(14) volatility filter.
# Uses 4h primary timeframe targeting 20-50 trades/year (80-200 total over 4 years).
# 1d EMA34 provides primary trend filter: bull when price > EMA34, bear when price < EMA34.
# Donchian(20) breakout provides institutional structure with proven edge.
# ATR(14) > 1.5x 50-period average filters for sufficient volatility.
# Position size 0.25 for balance between return and drawdown control.
# Discrete levels (0.0, ±0.25) minimize fee churn.
# Works in both bull (breakouts with trend) and bear (volatility-filtered mean reversion at channels).

name = "4h_Donchian20_1dEMA34_Trend_ATRVolFilter_v1"
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
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    vol_filter = atr_14 > 1.5 * atr_ma_50
    
    # Calculate Donchian channels (20-period)
    donchian_h = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_l = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient history for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(donchian_h[i]) or
            np.isnan(donchian_l[i]) or
            np.isnan(atr_14[i]) or
            np.isnan(atr_ma_50[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: 1d EMA34 direction
        price_above_ema = close[i] > ema_34_1d_aligned[i]
        price_below_ema = close[i] < ema_34_1d_aligned[i]
        
        # Donchian breakout conditions
        long_breakout = close[i] > donchian_h[i-1]  # Break above previous period high
        short_breakout = close[i] < donchian_l[i-1]  # Break below previous period low
        
        # Volatility filter: sufficient ATR for meaningful moves
        vol_ok = vol_filter[i]
        
        long_entry = price_above_ema and long_breakout and vol_ok
        short_entry = price_below_ema and short_breakout and vol_ok
        
        # Exit conditions: opposite Donchian level (mean reversion at channel)
        long_exit = close[i] < donchian_l[i]  # Exit long at lower channel
        short_exit = close[i] > donchian_h[i]  # Exit short at upper channel
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals