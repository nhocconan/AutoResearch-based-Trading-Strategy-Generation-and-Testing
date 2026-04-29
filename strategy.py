#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and ATR(14) stoploss
# Donchian channels capture long-term trend breakouts; weekly EMA34 filter ensures alignment with major trend
# ATR-based stoploss manages risk; discrete position sizing (0.25) minimizes fee churn
# Designed for ~10-25 trades/year on daily timeframe to avoid overtrading and fee drag
# Works in bull/bear via 1w EMA34 trend filter - only trades in direction of weekly momentum

name = "1d_Donchian20_1wEMA34_ATRStop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate ATR (14-period) for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period Donchian channels
    dc_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    dc_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_at_entry = 0.0
    
    start_idx = 20  # Donchian warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(dc_high[i]) or np.isnan(dc_low[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema34_1w = ema_34_1w_aligned[i]
        curr_atr = atr[i]
        curr_dc_high = dc_high[i]
        curr_dc_low = dc_low[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: stoploss hit or price breaks below Donchian low (failed breakout)
            if curr_close < entry_price - 2.5 * curr_atr or curr_close < curr_dc_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: stoploss hit or price breaks above Donchian high (failed breakout)
            if curr_close > entry_price + 2.5 * curr_atr or curr_close > curr_dc_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new breakout entries
            # Long breakout when price closes above Donchian high with 1w EMA34 uptrend
            if curr_close > curr_dc_high and curr_close > curr_ema34_1w:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                atr_at_entry = curr_atr
            # Short breakout when price closes below Donchian low with 1w EMA34 downtrend
            elif curr_close < curr_dc_low and curr_close < curr_ema34_1w:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                atr_at_entry = curr_atr
            else:
                signals[i] = 0.0
    
    return signals