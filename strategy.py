#!/usr/bin/env python3
"""
1d_1w_RSI_CCI_Confluence_Strategy
Hypothesis: RSI(14) and CCI(20) confluence on daily timeframe with weekly trend filter (EMA50) identifies high-probability reversals in both bull and bear markets.
Weekly EMA50 ensures we trade with the higher timeframe trend, while daily RSI < 30 and CCI < -100 identify oversold conditions for longs,
and RSI > 70 and CCI > 100 identify overbought conditions for shorts. Volume confirmation filters for institutional participation.
Target: 15-25 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate RSI(14) on daily closes
    def calculate_rsi(series, period):
        delta = np.diff(series)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full_like(series, np.nan)
        avg_loss = np.full_like(series, np.nan)
        
        # First average
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        
        # Wilder smoothing
        for i in range(period + 1, len(series)):
            avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close, 14)
    
    # Calculate CCI(20)
    def calculate_cci(high, low, close, period):
        tp = (high + low + close) / 3
        sma_tp = np.full_like(tp, np.nan)
        mad = np.full_like(tp, np.nan)
        
        for i in range(period - 1, len(tp)):
            sma_tp[i] = np.mean(tp[i - period + 1:i + 1])
            mad[i] = np.mean(np.abs(tp[i - period + 1:i + 1] - sma_tp[i]))
        
        cci = np.full_like(tp, np.nan)
        valid = (sma_tp != 0) & (~np.isnan(sma_tp)) & (~np.isnan(mad))
        cci[valid] = (tp[valid] - sma_tp[valid]) / (0.015 * mad[valid])
        return cci
    
    cci = calculate_cci(high, low, close, 20)
    
    # Volume confirmation: current volume > 1.3x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_confirmation = volume > (vol_ma_20 * 1.3)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(30, n):  # Start after indicators are ready
        # Skip if any required data is not ready
        if (np.isnan(rsi[i]) or np.isnan(cci[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. RSI < 30 (oversold)
        # 2. CCI < -100 (deeply oversold)
        # 3. Price above weekly EMA50 (bullish higher timeframe trend)
        # 4. Volume confirmation
        rsi_oversold = rsi[i] < 30
        cci_oversold = cci[i] < -100
        price_above_weekly_ema = close[i] > ema_50_1w_aligned[i]
        long_condition = rsi_oversold and cci_oversold and price_above_weekly_ema and volume_confirmation[i]
        
        # Short conditions:
        # 1. RSI > 70 (overbought)
        # 2. CCI > 100 (deeply overbought)
        # 3. Price below weekly EMA50 (bearish higher timeframe trend)
        # 4. Volume confirmation
        rsi_overbought = rsi[i] > 70
        cci_overbought = cci[i] > 100
        price_below_weekly_ema = close[i] < ema_50_1w_aligned[i]
        short_condition = rsi_overbought and cci_overbought and price_below_weekly_ema and volume_confirmation[i]
        
        if long_condition and position != 1:
            position = 1
            signals[i] = position_size
        elif short_condition and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "1d_1w_RSI_CCI_Confluence_Strategy"
timeframe = "1d"
leverage = 1.0