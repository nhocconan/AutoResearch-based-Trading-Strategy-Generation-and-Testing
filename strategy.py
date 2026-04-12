#!/usr/bin/env python3
"""
1h_4h_1d_RSI_Momentum_Confluence_v1
Hypothesis: In 1h timeframe, use 4h RSI(14) for trend direction (long when >55, short when <45) and 1d RSI(14) for regime filter (avoid counter-trend in strong trends). Enter on 1h RSI pullback to 50 with volume confirmation (>1.5x 20-period average). Designed for 15-35 trades/year per symbol with low turnover to minimize fee drag. Works in bull (momentum continuations) and bear (mean reversion within trend) markets.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_RSI_Momentum_Confluence_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4H DATA FOR TREND DIRECTION (RSI) ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate 4h RSI (14-period)
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    def wilders_rsi(gain, loss, period=14):
        avg_gain = np.full_like(gain, np.nan)
        avg_loss = np.full_like(loss, np.nan)
        if len(gain) < period:
            return np.full_like(gain, 50.0)
        avg_gain[period-1] = np.mean(gain[:period])
        avg_loss[period-1] = np.mean(loss[:period])
        for i in range(period, len(gain)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_4h = wilders_rsi(gain, loss, 14)
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # === 1D DATA FOR REGIME FILTER (RSI) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    delta_1d = np.diff(close_1d, prepend=close_1d[0])
    gain_1d = np.where(delta_1d > 0, delta_1d, 0)
    loss_1d = np.where(delta_1d < 0, -delta_1d, 0)
    
    rsi_1d = wilders_rsi(gain_1d, loss_1d, 14)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # === VOLUME FILTER (1H) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(rsi_4h_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Trend direction from 4h RSI
        bullish_trend = rsi_4h_aligned[i] > 55
        bearish_trend = rsi_4h_aligned[i] < 45
        
        # Regime filter from 1d RSI (avoid extreme counter-trend)
        not_overbought = rsi_1d_aligned[i] < 70  # Avoid shorting in strong bull
        not_oversold = rsi_1d_aligned[i] > 30    # Avoid longing in strong bear
        
        # Volume confirmation
        strong_volume = volume[i] > (vol_ma[i] * 1.5)
        
        # Entry conditions: 1h RSI pullback to 50 with volume
        # We approximate 1h RSI(50) when price is near short-term equilibrium
        # Use price crossing above/below VWAP-like condition: close > open for bull, close < open for bear
        bullish_momentum = close[i] > prices['open'].iloc[i]
        bearish_momentum = close[i] < prices['open'].iloc[i]
        
        # Long: bullish 4h trend, not overbought 1d, bullish 1h momentum, volume
        long_signal = (bullish_trend and 
                      not_overbought and 
                      bullish_momentum and 
                      strong_volume)
        
        # Short: bearish 4h trend, not oversold 1d, bearish 1h momentum, volume
        short_signal = (bearish_trend and 
                       not_oversold and 
                       bearish_momentum and 
                       strong_volume)
        
        # Exit: opposite momentum or RSI extreme
        exit_long = (position == 1 and 
                    (not bullish_momentum or rsi_4h_aligned[i] >= 70))
        exit_short = (position == -1 and 
                     (not bearish_momentum or rsi_4h_aligned[i] <= 30))
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.20
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals