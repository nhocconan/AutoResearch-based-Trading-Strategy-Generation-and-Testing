#!/usr/bin/env python3
# 6h_1d_ema_cross_volume_v1
# Hypothesis: 6h strategy using daily EMA crossover (EMA20/EMA50) with volume confirmation.
# Long: Price > daily EMA20 > daily EMA50, volume > 1.5x 20-period average, and close > open (bullish candle).
# Short: Price < daily EMA20 < daily EMA50, volume > 1.5x 20-period average, and close < open (bearish candle).
# Exit: Opposite EMA cross (EMA20 crosses EMA50 in opposite direction) or ATR trailing stop (2.0x ATR from extreme).
# Uses daily EMA for trend filter, volume to confirm participation, ATR for dynamic stops.
# Target: 12-37 trades/year (50-150 total over 4 years) on BTC/ETH/SOL.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_ema_cross_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for volatility filter and trailing stop
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift()).abs()
    tr3 = (low_s - close_s.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Get 1d data for EMA crossover (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate daily EMA20 and EMA50
    close_1d = pd.Series(df_1d['close'].values)
    ema20_1d = close_1d.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF EMA levels to 6h timeframe (wait for completed 1d bar)
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    long_high = 0.0   # highest high since long entry
    short_low = 0.0   # lowest low since short entry
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(ema20_1d_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or
            np.isnan(volume_ma[i]) or np.isnan(atr[i]) or np.isnan(close[i]) or np.isnan(high[i]) or np.isnan(low[i]) or
            np.isnan(open_price[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Trend filter: EMA20 > EMA50 for bullish bias, EMA20 < EMA50 for bearish bias
        ema_bullish = ema20_1d_aligned[i] > ema50_1d_aligned[i]
        ema_bearish = ema20_1d_aligned[i] < ema50_1d_aligned[i]
        
        # Candlestick filter: bullish/bearish candle
        bullish_candle = close[i] > open_price[i]
        bearish_candle = close[i] < open_price[i]
        
        if position == 1:  # Long position
            # Update highest high since entry
            long_high = max(long_high, high[i])
            # ATR trailing stop: exit if price drops 2.0*ATR from high
            if long_high > 0 and close[i] < long_high - 2.0 * atr[i]:
                position = 0
                long_high = 0.0
                signals[i] = 0.0
            # Exit: Opposite EMA cross (EMA20 crosses below EMA50)
            elif ema20_1d_aligned[i] < ema50_1d_aligned[i]:
                position = 0
                long_high = 0.0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            short_low = min(short_low, low[i])
            # ATR trailing stop: exit if price rises 2.0*ATR from low
            if short_low > 0 and close[i] > short_low + 2.0 * atr[i]:
                position = 0
                short_low = 0.0
                signals[i] = 0.0
            # Exit: Opposite EMA cross (EMA20 crosses above EMA50)
            elif ema20_1d_aligned[i] > ema50_1d_aligned[i]:
                position = 0
                short_low = 0.0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: EMA20 > EMA50, volume confirmed, and bullish candle
            if (ema_bullish and volume_confirmed and bullish_candle):
                position = 1
                long_high = high[i]
                signals[i] = 0.25
            # Short entry: EMA20 < EMA50, volume confirmed, and bearish candle
            elif (ema_bearish and volume_confirmed and bearish_candle):
                position = -1
                short_low = low[i]
                signals[i] = -0.25
    
    return signals