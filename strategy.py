#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour momentum strategy using daily RSI extremes with volume confirmation
# Long when daily RSI < 30 (oversold) and price closes above 4h VWAP
# Short when daily RSI > 70 (overbought) and price closes below 4h VWAP
# VWAP resets daily, providing intraday mean reversion edge
# Volume filter (>1.5x 20-period EMA) reduces false signals
# Designed for 20-40 trades/year, works in both bull (buy dips) and bear (sell rallies)
# Position size: 0.25

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data once for RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily RSI (14-period)
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Calculate 4h VWAP (typical price * volume) / cumulative volume
    typical_price = (high + low + close) / 3
    vwap_numerator = typical_price * volume
    vwap_denominator = volume
    
    # Reset VWAP at daily boundaries using date change
    dates = pd.to_datetime(prices['open_time']).date
    date_changes = np.concatenate([[True], dates[1:] != dates[:-1]])
    vwap_cumsum = np.where(date_changes, vwap_numerator, vwap_cumsum_prev + vwap_numerator) if 'vwap_cumsum_prev' in locals() else vwap_numerator
    vol_cumsum = np.where(date_changes, vwap_denominator, vol_cumsum_prev + vwap_denominator) if 'vol_cumsum_prev' in locals() else vwap_denominator
    vwap = vwap_cumsum / vol_cumsum
    
    # Store for next iteration
    vwap_cumsum_prev = vwap_cumsum
    vol_cumsum_prev = vol_cumsum
    
    # Volume moving average for confirmation (20-period EMA)
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    for i in range(20, n):
        # Get aligned daily RSI
        rsi_i = align_htf_to_ltf(prices, df_1d, rsi_1d)[i]
        
        if np.isnan(rsi_i) or np.isnan(vwap[i]) or np.isnan(vol_ma[i]):
            continue
        
        # Volume confirmation (1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Long: Daily RSI oversold + price above VWAP + volume
        if position == 0 and rsi_i < 30 and close[i] > vwap[i] and volume_confirm:
            position = 1
            signals[i] = position_size
        # Short: Daily RSI overbought + price below VWAP + volume
        elif position == 0 and rsi_i > 70 and close[i] < vwap[i] and volume_confirm:
            position = -1
            signals[i] = -position_size
        # Exit: Price returns to VWAP or RSI returns to neutral zone
        elif position != 0:
            if position == 1 and (close[i] < vwap[i] or rsi_i > 50):
                position = 0
                signals[i] = 0.0
            elif position == -1 and (close[i] > vwap[i] or rsi_i < 50):
                position = 0
                signals[i] = 0.0
    
    return signals

name = "4h_DailyRSI_VWAP_Momentum"
timeframe = "4h"
leverage = 1.0