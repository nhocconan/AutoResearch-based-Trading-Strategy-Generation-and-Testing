#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Supertrend for trend direction and 1d RSI extremes for mean reversion entries.
# In bull markets: 4h Supertrend up + 1d RSI < 30 = long (pullback in uptrend).
# In bear markets: 4h Supertrend down + 1d RSI > 70 = short (bounce in downtrend).
# Uses session filter (08-20 UTC) to reduce noise and ATR-based stoploss for risk control.
# Target: 15-30 trades/year by using tight 1d RSI extremes and 4h trend alignment.

name = "1h_Supertrend4h_RSI1dExtremes_Session_ATRStop_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 4h data ONCE before loop for Supertrend trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 10:
        return np.zeros(n)
    
    # Calculate Supertrend on 4h data (ATR=10, mult=3.0)
    hl2_4h = (df_4h['high'].values + df_4h['low'].values) / 2
    tr1_4h = df_4h['high'].values[1:] - df_4h['low'].values[1:]
    tr2_4h = np.abs(df_4h['high'].values[1:] - df_4h['close'].values[:-1])
    tr3_4h = np.abs(df_4h['low'].values[1:] - df_4h['close'].values[:-1])
    tr_4h = np.concatenate([[np.nan], np.maximum(tr1_4h, np.maximum(tr2_4h, tr3_4h))])
    atr_4h = pd.Series(tr_4h).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    up_4h = hl2_4h - (3.0 * atr_4h)
    down_4h = hl2_4h + (3.0 * atr_4h)
    
    supertrend_4h = np.full_like(hl2_4h, np.nan, dtype=float)
    direction_4h = np.ones_like(hl2_4h, dtype=int)  # 1 for uptrend, -1 for downtrend
    
    supertrend_4h[0] = up_4h[0]
    direction_4h[0] = 1
    
    for i in range(1, len(hl2_4h)):
        if close_4h := df_4h['close'].values[i]:
            if close_4h > supertrend_4h[i-1]:
                supertrend_4h[i] = up_4h[i]
                direction_4h[i] = 1
            else:
                supertrend_4h[i] = down_4h[i]
                direction_4h[i] = -1
        else:
            # Handle case where close_4h might be undefined (use typical price)
            typical_4h = hl2_4h[i]
            if typical_4h > supertrend_4h[i-1]:
                supertrend_4h[i] = up_4h[i]
                direction_4h[i] = 1
            else:
                supertrend_4h[i] = down_4h[i]
                direction_4h[i] = -1
    
    # Align Supertrend direction to 1h timeframe
    direction_4h_aligned = align_htf_to_ltf(prices, df_4h, direction_4h)
    
    # Load 1d data ONCE before loop for RSI extremes
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate RSI(14) on 1d data
    delta_1d = pd.Series(df_1d['close'].values).diff()
    gain_1d = delta_1d.clip(lower=0)
    loss_1d = -delta_1d.clip(upper=0)
    avg_gain_1d = gain_1d.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss_1d = loss_1d.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs_1d = avg_gain_1d / avg_loss_1d
    rsi_1d = 100 - (100 / (1 + rs_1d))
    rsi_1d_values = rsi_1d.values
    
    # Align RSI to 1h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d_values)
    
    # Calculate ATR(14) for 1h timeframe stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 100  # warmup for Supertrend, RSI, and ATR
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            # Outside session: flatten position if any
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_direction = direction_4h_aligned[i]
        curr_rsi = rsi_1d_aligned[i]
        curr_atr = atr[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: 4h uptrend + 1d RSI < 30 (oversold)
            if (curr_direction == 1 and 
                curr_rsi < 30):
                signals[i] = 0.20
                position = 1
                entry_price = curr_close
            # Short: 4h downtrend + 1d RSI > 70 (overbought)
            elif (curr_direction == -1 and 
                  curr_rsi > 70):
                signals[i] = -0.20
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit conditions: 4h trend turns down OR stoploss hit
            if (curr_direction == -1 or 
                curr_close < entry_price - 1.5 * curr_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit conditions: 4h trend turns up OR stoploss hit
            if (curr_direction == 1 or 
                curr_close > entry_price + 1.5 * curr_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals