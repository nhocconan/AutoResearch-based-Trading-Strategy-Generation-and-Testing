#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_ema_cross_trend_filter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # 1w EMA 50 for long-term trend
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # 1w ATR for volatility filter
    tr1 = df_1w['high'].values[1:] - df_1w['low'].values[1:]
    tr2 = np.abs(df_1w['high'].values[1:] - df_1w['close'].values[:-1])
    tr3 = np.abs(df_1w['low'].values[1:] - df_1w['close'].values[:-1])
    tr_1w = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    
    # 1d EMA 21 for entry signal
    ema_21_1d = pd.Series(close).ewm(span=21, min_periods=21, adjust=False).mean().values
    
    # 1d ATR for stop loss
    tr1_d = high[1:] - low[1:]
    tr2_d = np.abs(high[1:] - close[:-1])
    tr3_d = np.abs(low[1:] - close[:-1])
    tr_d = np.concatenate([[np.nan], np.maximum(tr1_d, np.maximum(tr2_d, tr3_d))])
    atr_d = pd.Series(tr_d).rolling(window=14, min_periods=14).mean().values
    
    # 1d volume filter: volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1w indicators to daily timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr_1w_aligned[i]) or
            np.isnan(ema_21_1d[i]) or np.isnan(atr_d[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        atr_val = atr_d[i]
        
        ema_50_1w_val = ema_50_1w_aligned[i]
        ema_21_1d_val = ema_21_1d[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.3 * vol_ma
        
        # Volatility filter: avoid extremely low volatility
        vol_filter = atr_val > 0.003 * price_close  # ATR > 0.3% of price
        
        # Long conditions: price above 21 EMA, above 1w 50 EMA trend, with volume and volatility
        long_signal = volume_confirmed and vol_filter and (price_close > ema_21_1d_val) and (price_close > ema_50_1w_val)
        
        # Short conditions: price below 21 EMA, below 1w 50 EMA trend, with volume and volatility
        short_signal = volume_confirmed and vol_filter and (price_close < ema_21_1d_val) and (price_close < ema_50_1w_val)
        
        # Stop loss: 2 * ATR from entry
        if position == 1 and price_low < ema_21_1d_val - 2.0 * atr_val:
            position = 0
            signals[i] = 0.0
        elif position == -1 and price_high > ema_21_1d_val + 2.0 * atr_val:
            position = 0
            signals[i] = 0.0
        # Exit when price crosses back below/above 21 EMA (mean reversion)
        elif position == 1 and price_close < ema_21_1d_val:
            position = 0
            signals[i] = 0.0
        elif position == -1 and price_close > ema_21_1d_val:
            position = 0
            signals[i] = 0.0
        # Entry signals
        elif long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Daily EMA(21) cross with weekly EMA(50) trend filter captures medium-term trends
# while avoiding whipsaws. Volume and volatility filters ensure quality entries.
# Long when price > EMA21 and > weekly EMA50 (bullish alignment).
# Short when price < EMA21 and < weekly EMA50 (bearish alignment).
# Stop loss at 2x ATR from EMA21 to limit drawdown.
# Exit when price crosses back below/above EMA21 (mean reversion within trend).
# Designed for 15-25 trades per year on daily timeframe, balancing opportunity and fee cost.
# Works in bull markets (riding uptrends) and bear markets (riding downtrends).
# Weekly trend filter prevents counter-trend trading during strong moves.