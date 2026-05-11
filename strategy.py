#!/usr/bin/env python3
"""
6h_1w_EMA50_Trend_RSI14_Entry
Hypothesis: Uses weekly EMA50 to determine long-term trend (bull/bear), with RSI(14) on 6h for entry timing.
In bull trend (price > weekly EMA50), go long when RSI crosses above 30 (oversold bounce).
In bear trend (price < weekly EMA50), go short when RSI crosses below 70 (overbought rejection).
Requires volume confirmation (volume > 1.5x 24-bar average) to avoid low-quality signals.
Designed to work in both bull and bear markets by following higher-timeframe trend.
Targets low trade frequency (12-37/year) via weekly trend filter and RSI entry signals.
"""

name = "6h_1w_EMA50_Trend_RSI14_Entry"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_rsi(close, period=14):
    """Calculate Relative Strength Index"""
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50).values  # neutral RSI when no loss

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Weekly EMA50 for Trend Filter ---
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_6h = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # --- Daily RSI(14) for Entry Timing ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    rsi_1d = calculate_rsi(df_1d['close'].values, 14)
    rsi_1d_6h = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # --- Volume Spike Detection (24-period average on 6h) ---
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_6h[i]) or 
            np.isnan(rsi_1d_6h[i]) or
            np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Trend determination
        bull_trend = close[i] > ema_50_1w_6h[i]
        bear_trend = close[i] < ema_50_1w_6h[i]
        
        # RSI signals (using previous bar to avoid look-ahead)
        rsi_prev = rsi_1d_6h[i-1] if i > 0 else 50
        rsi_cross_up = (rsi_1d_6h[i] > 30) & (rsi_prev <= 30)  # RSI crosses above 30
        rsi_cross_down = (rsi_1d_6h[i] < 70) & (rsi_prev >= 70)  # RSI crosses below 70
        
        # Volume confirmation threshold
        volume_spike = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: bull trend + RSI crosses above 30 + volume
            if bull_trend and rsi_cross_up and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: bear trend + RSI crosses below 70 + volume
            elif bear_trend and rsi_cross_down and volume_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: trend reversal or opposite RSI extreme
            if position == 1:
                # Exit long: bear trend OR RSI crosses below 30 (failed bounce)
                if bear_trend or (rsi_1d_6h[i] < 30 and rsi_prev >= 30):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: bull trend OR RSI crosses above 70 (failed rejection)
                if bull_trend or (rsi_1d_6h[i] > 70 and rsi_prev <= 70):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals