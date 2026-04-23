#!/usr/bin/env python3
"""
Hypothesis: 1h volume-weighted RSI mean reversion with 4h trend filter.
- Primary timeframe: 1h for precise entry timing
- HTF: 4h EMA50 for trend direction (avoid counter-trend trades)
- Entry: RSI(14) < 30 for long, RSI(14) > 70 for short (mean reversion extremes)
- Volume filter: only trade when volume > 1.2x 20-period average (avoid low-liquidity false signals)
- Exit: RSI returns to neutral zone (40-60 range) or opposite extreme
- Position size: 0.20 (20% of capital) to limit drawdown in volatile markets
- Session filter: 08-20 UTC to avoid Asian session noise and focus on active London/NY overlap
- Target: 15-30 trades/year (60-120 total over 4 years) to minimize fee drag
- RSI mean reversion works in both bull (buy dips) and bear (sell rallies) markets
- Volume confirmation ensures participation, reducing whipsaws in ranging conditions
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) - open_time is already datetime64[ms]
    hours = prices.index.hour  # prices.index is DatetimeIndex
    
    # Volume confirmation: > 1.2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # RSI(14) calculation
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral when undefined
    
    # Load 4h data ONCE before loop for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14, 50)  # volume MA, RSI, 4h EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        # Volume spike confirmation (> 1.2x average)
        volume_ok = volume[i] > 1.2 * vol_ma[i]
        
        if position == 0 and in_session and volume_ok:
            # Long entry: RSI < 30 (oversold) + price above 4h EMA50 (uptrend bias)
            if rsi[i] < 30 and close[i] > ema_50_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short entry: RSI > 70 (overbought) + price below 4h EMA50 (downtrend bias)
            elif rsi[i] > 70 and close[i] < ema_50_aligned[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: RSI returns to neutral (>= 40) or breaks downtrend
            if rsi[i] >= 40 or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: RSI returns to neutral (<= 60) or breaks uptrend
            if rsi[i] <= 60 or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_VolumeWeighted_RSI_MeanReversion_4hEMA50_Trend"
timeframe = "1h"
leverage = 1.0