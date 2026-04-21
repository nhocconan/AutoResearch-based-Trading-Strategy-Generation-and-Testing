#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h 10-period RSI combined with 1d ADX trend filter and volume confirmation.
# Long when RSI(10) < 30 (oversold) and 1d ADX > 25 (trending) and volume > 1.5x 20-period average.
# Short when RSI(10) > 70 (overbought) and 1d ADX > 25 (trending) and volume > 1.5x 20-period average.
# Exit when RSI returns to neutral zone (40-60).
# Designed to capture mean reversion within strong trends, avoiding choppy markets.
# Target: 12-37 trades/year by requiring RSI extremes + trend alignment + volume confirmation.

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Load 1d for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily ADX(14)
    high = df_1d['high'].values
    low = df_1d['low'].values
    close = df_1d['close'].values
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
    def wilders_smoothing(x, period):
        result = np.full_like(x, np.nan)
        if len(x) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(x[1:period])
        # Subsequent values: smoothed = previous * (1 - 1/period) + current * (1/period)
        for i in range(period, len(x)):
            if not np.isnan(result[i-1]) and not np.isnan(x[i]):
                result[i] = result[i-1] * (1 - 1/period) + x[i] * (1/period)
            else:
                result[i] = np.nan
        return result
    
    atr = wilders_smoothing(tr, 14)
    plus_di = 100 * wilders_smoothing(plus_dm, 14) / atr
    minus_di = 100 * wilders_smoothing(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilders_smoothing(dx, 14)
    
    # Align 1d ADX to 12h
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 12h RSI(10)
    close_prices = prices['close'].values
    delta = np.diff(close_prices)
    delta = np.concatenate([[np.nan], delta])
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    avg_gain = wilders_smoothing(gain, 10)
    avg_loss = wilders_smoothing(loss, 10)
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if data not ready
        if np.isnan(adx_1d_aligned[i]) or np.isnan(rsi[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Calculate 20-period volume average
        vol_lookback_start = max(0, i - 19)
        vol_window = prices['volume'].iloc[vol_lookback_start:i+1].values
        vol_ma_20 = np.mean(vol_window)
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume > 1.5 * vol_ma_20
        
        # Trend filter: 1d ADX > 25 indicates trending market
        trending = adx_1d_aligned[i] > 25
        
        if position == 0:
            # Enter long on RSI oversold (<30) in trending market with volume confirmation
            if rsi[i] < 30 and trending and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Enter short on RSI overbought (>70) in trending market with volume confirmation
            elif rsi[i] > 70 and trending and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: RSI returns to neutral zone (40-60)
            exit_signal = False
            
            if position == 1:
                # Exit long when RSI >= 40 (recovering from oversold)
                if rsi[i] >= 40:
                    exit_signal = True
            elif position == -1:
                # Exit short when RSI <= 60 (declining from overbought)
                if rsi[i] <= 60:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_RSI10_ADX25_Volume"
timeframe = "12h"
leverage = 1.0