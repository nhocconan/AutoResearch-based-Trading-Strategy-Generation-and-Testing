#!/usr/bin/env python3
"""
Hypothesis: 1h strategy with 4h/1d multi-timeframe confirmation.
- Trend: 1d close > 1d EMA200 = bull, < = bear (avoids counter-trend trades)
- Entry: In bull trend, buy when 1h RSI < 30 and price > 4h VWAP (pullback to value)
         In bear trend, sell when 1h RSI > 70 and price < 4h VWAP (bounce to value)
- Exit: RSI crosses back to neutral (40-60 range) or trend flips
- Volume: Require 1h volume > 20-period average for confirmation
- Position size: 0.20 (20%)
- Session filter: 08-20 UTC (avoid low liquidity hours)
Target: 15-30 trades/year (60-120 over 4 years) - conservative to avoid fee drag
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_rsi_vwap_4h1d_trend_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Session filter: 08-20 UTC (pre-market to post-US close)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # === 1D TREND FILTER ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    one_d_close = df_1d['close'].values
    one_d_ema = pd.Series(one_d_close).ewm(span=200, adjust=False, min_periods=200).mean().values
    one_d_ema_aligned = align_htf_to_ltf(prices, df_1d, one_d_ema)
    
    # === 4H VWAP (Volume Weighted Average Price) ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) == 0:
        return np.zeros(n)
    # Typical price for VWAP
    tp_4h = (df_4h['high'] + df_4h['low'] + df_4h['close']) / 3
    vwap_4h = (tp_4h * df_4h['volume']).cumsum() / df_4h['volume'].cumsum()
    vwap_4h_aligned = align_htf_to_ltf(prices, df_4h, vwap_4h.values)
    
    # === 1H RSI (14) ===
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # === 1H VOLUME MA (20) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):  # Start after 1d EMA warmup
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        if np.isnan(one_d_ema_aligned[i]) or np.isnan(vwap_4h_aligned[i]) or \
           np.isnan(rsi[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        bull_trend = close[i] > one_d_ema_aligned[i]
        
        if position == 1:  # Long position
            # Exit: RSI > 50 (taking profit) OR trend turns bearish
            if rsi[i] > 50 or not bull_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: RSI < 50 (taking profit) OR trend turns bullish
            if rsi[i] < 50 or bull_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Need volume confirmation
            if volume[i] <= vol_ma[i]:
                signals[i] = 0.0
                continue
            
            # Entry logic based on trend and RSI extremes
            if bull_trend:
                # In bull market: buy pullbacks (RSI < 30) to 4h VWAP support
                if rsi[i] < 30 and close[i] > vwap_4h_aligned[i]:
                    position = 1
                    signals[i] = 0.20
            else:
                # In bear market: sell bounces (RSI > 70) to 4h VWAP resistance
                if rsi[i] > 70 and close[i] < vwap_4h_aligned[i]:
                    position = -1
                    signals[i] = -0.20
    
    return signals