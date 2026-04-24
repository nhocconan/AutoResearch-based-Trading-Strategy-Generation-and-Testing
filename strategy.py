#!/usr/bin/env python3
"""
Hypothesis: 1h RSI mean reversion with 4h trend filter and volume spike confirmation.
- Primary timeframe: 1h for precise entry/exit timing.
- HTF: 4h EMA50 for trend direction (bullish if close > EMA50, bearish if close < EMA50).
- Volume: Current 1h volume > 2.0 * 24-period volume MA to capture institutional interest.
- RSI: 14-period RSI for mean reversion signals.
- Entry: Long when RSI < 30 AND 4h EMA50 bullish AND volume spike.
         Short when RSI > 70 AND 4h EMA50 bearish AND volume spike.
- Exit: Opposite RSI level (RSI > 70 for long, RSI < 30 for short) or loss of volume confirmation.
- Signal size: 0.20 discrete to limit drawdown and reduce fee churn.
- Session filter: Only trade between 08:00-20:00 UTC to avoid low-liquidity periods.
- Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.
This combines mean reversion in ranging markets with trend filtering to avoid counter-trend trades,
while volume spikes confirm institutional participation. Works in both bull and bear markets by
only taking trades in the direction of the 4h trend.
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
    
    # Calculate 1h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    # Wilder's smoothing (equivalent to EMA with alpha=1/period)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50
    df_4h_close = df_4h['close'].values
    ema_4h = pd.Series(df_4h_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 24-period 4h volume MA
    df_4h_volume = df_4h['volume'].values
    vol_ma_4h = pd.Series(df_4h_volume).rolling(window=24, min_periods=24).mean().values
    
    # Align HTF indicators to 1h
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    # Volume confirmation: current 1h volume > 2.0 * 24-period 4h volume MA (aligned)
    volume_spike = volume > (2.0 * vol_ma_4h_aligned)
    
    # Session filter: 08:00-20:00 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 24, 14)  # Need enough bars for EMA50, volume MA, and RSI
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(rsi[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_rsi = rsi[i]
        ema_val = ema_4h_aligned[i]
        
        if position == 0:
            # Check for entry signals with volume spike and session filter
            if volume_spike[i]:
                # Bullish: RSI < 30 (oversold) AND 4h EMA50 bullish (close > EMA)
                if curr_rsi < 30.0 and curr_close > ema_val:
                    signals[i] = 0.20
                    position = 1
                # Bearish: RSI > 70 (overbought) AND 4h EMA50 bearish (close < EMA)
                elif curr_rsi > 70.0 and curr_close < ema_val:
                    signals[i] = -0.20
                    position = -1
        elif position == 1:
            # Long exit: RSI > 70 (overbought) OR loss of volume confirmation OR outside session
            if curr_rsi > 70.0 or not volume_spike[i] or not in_session[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: RSI < 30 (oversold) OR loss of volume confirmation OR outside session
            if curr_rsi < 30.0 or not volume_spike[i] or not in_session[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_RSI_MeanReversion_4hEMA50_Trend_VolumeSpike_Session_v1"
timeframe = "1h"
leverage = 1.0