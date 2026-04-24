#!/usr/bin/env python3
"""
Hypothesis: 1h RSI(14) mean reversion with 4h EMA(50) trend filter and 1d volume spike confirmation.
- Primary timeframe: 1h for precise entry/exit timing.
- HTF: 4h EMA(50) for trend direction (bullish if close > EMA50, bearish if close < EMA50).
- Volume: Current 1h volume > 2.0 * 20-period 1d volume MA to avoid low-momentum breakouts.
- Entry: Long when RSI(14) < 30 AND 4h EMA50 trend bullish AND volume spike.
         Short when RSI(14) > 70 AND 4h EMA50 trend bearish AND volume spike.
- Exit: RSI crosses back to neutral (40 for long exit, 60 for short exit) OR loss of volume confirmation.
- Signal size: 0.20 discrete to limit drawdown and reduce fee churn.
- Session filter: 08-20 UTC to avoid low-liquidity hours.
- Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate RSI(14) on 1h
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # Neutral RSI when undefined
    
    # Get 4h data for EMA(50) trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 4h close
    ema_50 = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 1d data for volume MA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-period volume MA on 1d
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 1h
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Volume confirmation: current 1h volume > 2.0 * 20-period 1d volume MA (aligned)
    volume_spike = volume > (2.0 * vol_ma_1d_aligned)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # Already datetime64[ns], .hour works
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 14)  # Need enough bars for EMA50, vol MA, RSI
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema_50_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(volume_spike[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_50_val = ema_50_aligned[i]
        curr_close = close[i]
        curr_rsi = rsi[i]
        
        if position == 0:
            # Check for entry signals with volume spike and session
            if volume_spike[i]:
                # Bullish mean reversion: RSI oversold AND 4h EMA50 bullish (close > EMA50)
                if curr_rsi < 30 and curr_close > ema_50_val:
                    signals[i] = 0.20
                    position = 1
                # Bearish mean reversion: RSI overbought AND 4h EMA50 bearish (close < EMA50)
                elif curr_rsi > 70 and curr_close < ema_50_val:
                    signals[i] = -0.20
                    position = -1
        elif position == 1:
            # Long exit: RSI crosses above 40 OR loss of volume confirmation OR outside session
            if curr_rsi > 40 or not volume_spike[i] or not in_session[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: RSI crosses below 60 OR loss of volume confirmation OR outside session
            if curr_rsi < 60 or not volume_spike[i] or not in_session[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_RSI14_4hEMA50Trend_1dVolumeSpike_Session_v1"
timeframe = "1h"
leverage = 1.0