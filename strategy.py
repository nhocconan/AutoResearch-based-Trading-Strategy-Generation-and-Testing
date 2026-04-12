#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Elder Ray + regime filter
    # Bull Power (close - EMA13) and Bear Power (EMA13 - close) with ADX regime
    # Long when Bull Power > 0, ADX > 20, and +DI > -DI
    # Short when Bear Power > 0, ADX > 20, and -DI > +DI
    # Uses 12h EMA200 for major trend filter to avoid counter-trend trades
    # Designed for low trade frequency (target: 12-37/year) to minimize fee drag
    # Works in bull/bear via ADX regime and trend filter
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Elder Ray components: EMA13 for Bull/Bear Power
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = close - ema13  # Bull Power = Close - EMA13
    bear_power = ema13 - close  # Bear Power = EMA13 - Close
    
    # Calculate ADX and DI components for regime filter
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # First bar has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed TR, +DM, -DM (using Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        """Wilder's smoothing (equivalent to EMA with alpha=1/period)"""
        alpha = 1.0 / period
        result = np.full_like(data, np.nan)
        # First value is simple average
        if len(data) >= period:
            result[period-1] = np.mean(data[:period])
        # Subsequent values via Wilder's smoothing
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]):
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    atr = wilders_smoothing(tr, 14)
    plus_dm_smooth = wilders_smoothing(plus_dm, 14)
    minus_dm_smooth = wilders_smoothing(minus_dm, 14)
    
    # Avoid division by zero
    plus_di = np.where(atr != 0, 100 * plus_dm_smooth / atr, 0)
    minus_di = np.where(atr != 0, 100 * minus_dm_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = wilders_smoothing(dx, 14)
    
    # Get 12h data for EMA200 major trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 200:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema200_12h = pd.Series(close_12h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema200_12h)
    
    # Volume confirmation: volume > 1.3 * 20-period average (to reduce false signals)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(adx[i]) or np.isnan(plus_di[i]) or np.isnan(minus_di[i]) or 
            np.isnan(ema200_12h_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine 12h major trend
        bullish_major_trend = close[i] > ema200_12h_aligned[i]
        bearish_major_trend = close[i] < ema200_12h_aligned[i]
        
        # Entry conditions
        long_entry = False
        short_entry = False
        
        # Long: Bull Power positive, ADX > 20 (trending), +DI > -DI, bullish major trend, volume confirmation
        if (bull_power[i] > 0 and adx[i] > 20 and plus_di[i] > minus_di[i] and 
            bullish_major_trend and volume_spike[i]):
            long_entry = True
        
        # Short: Bear Power positive, ADX > 20 (trending), -DI > +DI, bearish major trend, volume confirmation
        if (bear_power[i] > 0 and adx[i] > 20 and minus_di[i] > plus_di[i] and 
            bearish_major_trend and volume_spike[i]):
            short_entry = True
        
        # Exit conditions: opposite power signal or ADX weakening (range market)
        long_exit = (bear_power[i] > 0) or (adx[i] < 15)  # Bear Power positive or weak trend
        short_exit = (bull_power[i] > 0) or (adx[i] < 15)  # Bull Power positive or weak trend
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_12h_elder_ray_adx_trend_v1"
timeframe = "6h"
leverage = 1.0