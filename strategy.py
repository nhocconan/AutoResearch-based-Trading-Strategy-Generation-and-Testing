#!/usr/bin/env python3
"""
12h_1wk_momentum_reversion_v1
Hypothesis: Use 1-week momentum (RSI) to identify overbought/oversold conditions,
combined with 12h price action near weekly support/resistance levels.
In ranging markets (weekly RSI between 30-70), fade extreme 12h moves.
In trending markets (weekly RSI >70 or <30), trade pullbacks to weekly VWAP.
Volume confirmation filters weak signals. Works in both bull/bear via regime adaptation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1wk_momentum_reversion_v1"
timeframe = "12h"
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
    
    # Weekly data for momentum and context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Weekly RSI(14) for momentum regime
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1w = 100 - (100 / (1 + rs))
    
    # Weekly VWAP for dynamic support/resistance
    typical_price_1w = (df_1w['high'].values + df_1w['low'].values + df_1w['close'].values) / 3
    vwap_num = (typical_price_1w * df_1w['volume'].values).cumsum()
    vwap_den = df_1w['volume'].values.cumsum()
    vwap_1w = vwap_num / (vwap_den + 1e-10)
    
    # Weekly ATR for volatility normalization
    high_low_1w = df_1w['high'].values - df_1w['low'].values
    high_close_1w = np.abs(df_1w['high'].values - np.roll(close_1w, 1))
    low_close_1w = np.abs(df_1w['low'].values - np.roll(close_1w, 1))
    high_close_1w[0] = 0
    low_close_1w[0] = 0
    tr_1w = np.maximum(high_low_1w, np.maximum(high_close_1w, low_close_1w))
    atr_1w = pd.Series(tr_1w).ewm(span=14, adjust=False).mean().values
    
    # Align weekly indicators to 12h timeframe
    rsi_1w_12h = align_htf_to_ltf(prices, df_1w, rsi_1w)
    vwap_1w_12h = align_htf_to_ltf(prices, df_1w, vwap_1w)
    atr_1w_12h = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # 12h RSI for entry timing
    delta_12h = np.diff(close, prepend=close[0])
    gain_12h = np.where(delta_12h > 0, delta_12h, 0)
    loss_12h = np.where(delta_12h < 0, -delta_12h, 0)
    avg_gain_12h = pd.Series(gain_12h).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss_12h = pd.Series(loss_12h).ewm(alpha=1/14, adjust=False).mean().values
    rs_12h = avg_gain_12h / (avg_loss_12h + 1e-10)
    rsi_12h = 100 - (100 / (1 + rs_12h))
    
    # 12h volume filter
    vol_ma = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(rsi_1w_12h[i]) or np.isnan(vwap_1w_12h[i]) or 
            np.isnan(atr_1w_12h[i]) or np.isnan(rsi_12h[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x average
        vol_confirm = volume[i] > 1.3 * vol_ma[i]
        
        # Distance from weekly VWAP in ATR units
        if atr_1w_12h[i] > 0:
            vwap_dist = (close[i] - vwap_1w_12h[i]) / atr_1w_12h[i]
        else:
            vwap_dist = 0
        
        if position == 1:  # Long position
            # Exit: RSI overextended or price too far above VWAP
            if (rsi_12h[i] > 75 or vwap_dist > 2.0):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: RSI oversold or price too far below VWAP
            if (rsi_12h[i] < 25 or vwap_dist < -2.0):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Regime: weekly RSI > 60 = bullish bias, < 40 = bearish bias
            if rsi_1w_12h[i] > 60:  # Bullish regime
                # Look for pullbacks to VWAP with momentum divergence
                if (vwap_dist < -0.5 and  # Below VWAP
                    rsi_12h[i] < 40 and   # Oversold on 12h
                    vol_confirm):
                    position = 1
                    signals[i] = 0.25
            elif rsi_1w_12h[i] < 40:  # Bearish regime
                # Look for bounces off VWAP with momentum divergence
                if (vwap_dist > 0.5 and   # Above VWAP
                    rsi_12h[i] > 60 and   # Overbought on 12h
                    vol_confirm):
                    position = -1
                    signals[i] = -0.25
            else:  # Neutral regime (30-70) - mean reversion
                # Fade extreme deviations from VWAP
                if (vwap_dist < -1.5 and  # Far below VWAP
                    vol_confirm):
                    position = 1
                    signals[i] = 0.25
                elif (vwap_dist > 1.5 and  # Far above VWAP
                      vol_confirm):
                    position = -1
                    signals[i] = -0.25
    
    return signals