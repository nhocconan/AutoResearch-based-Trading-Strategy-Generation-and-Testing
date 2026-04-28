#!/usr/bin/env python3
"""
1d_ADX_Trend_RSI_MeanReversion
Hypothesis: On daily timeframe, use ADX to detect trend regimes. When ADX > 25 (trending), trade pullbacks to EMA21 in trend direction. When ADX < 20 (range), trade RSI extremes for mean reversion. Volume confirmation filters false signals. Designed to work in both bull and bear markets by adapting to regime.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ADX(14) for regime detection
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(arr[:period]) / period
        # Subsequent values: smoothed = prev * (1 - 1/period) + current * (1/period)
        for i in range(period, len(arr)):
            if not np.isnan(result[i-1]):
                result[i] = result[i-1] * (1 - 1/period) + arr[i] * (1/period)
        return result
    
    tr_smooth = wilder_smooth(tr, 14)
    plus_dm_smooth = wilder_smooth(plus_dm, 14)
    minus_dm_smooth = wilder_smooth(minus_dm, 14)
    
    # Avoid division by zero
    plus_di = np.where(tr_smooth != 0, plus_dm_smooth / tr_smooth * 100, 0)
    minus_di = np.where(tr_smooth != 0, minus_dm_smooth / tr_smooth * 100, 0)
    dx = np.where((plus_di + minus_di) != 0, np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 0)
    adx = wilder_smooth(dx, 14)
    
    # EMA21 for trend following pullbacks
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # RSI(14) for mean reversion in ranging markets
    def calculate_rsi(prices, period=14):
        delta = np.diff(prices)
        delta = np.insert(delta, 0, 0)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full_like(prices, np.nan, dtype=float)
        avg_loss = np.full_like(prices, np.nan, dtype=float)
        
        if len(prices) < period:
            return avg_gain
            
        # First average
        avg_gain[period-1] = np.mean(gain[:period])
        avg_loss[period-1] = np.mean(loss[:period])
        
        # Subsequent averages
        for i in range(period, len(prices)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i]) / period
            
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close, 14)
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx[i]) or np.isnan(ema_21[i]) or np.isnan(rsi[i]) or 
            np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Regime-based logic
        if adx[i] > 25:  # Trending regime
            # Long: pullback to EMA21 in uptrend
            long_entry = (close[i] > ema_21[i] and 
                         close[i] < ema_21[i] * 1.02 and  # Within 2% above EMA
                         volume_surge[i])
            # Short: pullback to EMA21 in downtrend
            short_entry = (close[i] < ema_21[i] and 
                          close[i] > ema_21[i] * 0.98 and  # Within 2% below EMA
                          volume_surge[i])
        elif adx[i] < 20:  # Ranging regime
            # Long: RSI oversold
            long_entry = (rsi[i] < 30 and volume_surge[i])
            # Short: RSI overbought
            short_entry = (rsi[i] > 70 and volume_surge[i])
        else:  # Transition zone - no trades
            long_entry = False
            short_entry = False
        
        # Exit conditions
        long_exit = (position == 1 and 
                    (adx[i] < 20 or  # Regime changed to range
                     rsi[i] > 70 or  # RSI overbought
                     close[i] < ema_21[i] * 0.95))  # Stop: 5% below EMA
        short_exit = (position == -1 and 
                     (adx[i] < 20 or  # Regime changed to range
                      rsi[i] < 30 or  # RSI oversold
                      close[i] > ema_21[i] * 1.05))  # Stop: 5% above EMA
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_ADX_Trend_RSI_MeanReversion"
timeframe = "1d"
leverage = 1.0