#!/usr/bin/env python3
"""
Hypothesis: 1h price action combined with 4h/1d structure and volume confirmation.
- Primary timeframe: 1h for precise entry/exit timing within HTF structure.
- HTF trend: 4h EMA200 filters direction (bullish if close > EMA200, bearish if close < EMA200).
- HTF momentum: 1d RSI(14) avoids extreme overbought/oversold conditions that often fail in trends.
- Volume: Current 1h volume > 1.5 * 20-period volume MA to confirm participation.
- Entry timing: Long when price pulls back to 20-period EMA in uptrend; short when price rallies to 20-period EMA in downtrend.
- Exit: Opposite EMA touch or loss of volume/structure confirmation.
- Signal size: 0.20 discrete to limit drawdown and reduce fee churn.
- Session filter: Only trade between 08:00-20:00 UTC to avoid low-liquidity periods.
- Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.
This strategy buys the dip in strong uptrends and sells the rally in strong downtrends, using volume to confirm institutional participation and higher timeframes to avoid counter-trend trades. Works in bull markets by buying dips and in bear markets by selling rallies.
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
    
    # Calculate 1h EMA20 for dynamic support/resistance
    close_s = pd.Series(close)
    ema_20 = close_s.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Get 4h data for EMA200 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 200:
        return np.zeros(n)
    
    # Calculate 4h EMA200
    df_4h_close = df_4h['close'].values
    ema_4h_200 = pd.Series(df_4h_close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Get 1d data for RSI momentum filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d RSI(14)
    df_1d_close = df_1d['close'].values
    delta = np.diff(df_1d_close, prepend=df_1d_close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi_1d = 100.0 - (100.0 / (1.0 + rs))
    
    # Calculate 20-period 1h volume MA
    volume_s = pd.Series(volume)
    vol_ma_20 = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 1h
    ema_4h_200_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_200)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Volume confirmation: current 1h volume > 1.5 * 20-period volume MA
    volume_confirmed = volume > (1.5 * vol_ma_20)
    
    # Session filter: 08:00-20:00 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(200, 20, 14)  # Need enough bars for EMA200, EMA20, and 1d RSI
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema_4h_200_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(ema_20[i]) or np.isnan(volume_confirmed[i]) or 
            np.isnan(close[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_ema20 = ema_20[i]
        ema_trend = ema_4h_200_aligned[i]
        rsi_momentum = rsi_1d_aligned[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation and session filter
            if volume_confirmed[i]:
                # Bullish: Price near 20 EMA (support) in uptrend, 1d RSI not extremely overbought
                if (curr_close >= curr_ema20 * 0.995 and curr_close <= curr_ema20 * 1.005) and \
                   curr_close > ema_trend and \
                   rsi_momentum < 70:  # Avoid extremely overbought conditions
                    signals[i] = 0.20
                    position = 1
                # Bearish: Price near 20 EMA (resistance) in downtrend, 1d RSI not extremely oversold
                elif (curr_close >= curr_ema20 * 0.995 and curr_close <= curr_ema20 * 1.005) and \
                     curr_close < ema_trend and \
                     rsi_momentum > 30:  # Avoid extremely oversold conditions
                    signals[i] = -0.20
                    position = -1
        elif position == 1:
            # Long exit: Price crosses above 20 EMA (momentum) OR loss of volume/structure OR outside session
            if curr_close > curr_ema20 * 1.01 or not volume_confirmed[i] or curr_close <= ema_trend or not in_session[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: Price crosses below 20 EMA (momentum) OR loss of volume/structure OR outside session
            if curr_close < curr_ema20 * 0.99 or not volume_confirmed[i] or curr_close >= ema_trend or not in_session[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_EMA20_Pullback_4hEMA200_Trend_1dRSI_Momentum_Volume_Session_v1"
timeframe = "1h"
leverage = 1.0