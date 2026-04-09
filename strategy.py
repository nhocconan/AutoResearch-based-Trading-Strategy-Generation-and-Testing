#!/usr/bin/env python3
# 6h_1d_donchian_volume_pullback_v1
# Hypothesis: 6h strategy using daily Donchian channel breakout with volume confirmation and pullback entry.
# Long: Price breaks above daily Donchian(20) high, pulls back to touch or cross above the 20-period EMA on 6h, with volume > 1.5x 20-period average.
# Short: Price breaks below daily Donchian(20) low, pulls back to touch or cross below the 20-period EMA on 6h, with volume > 1.5x 20-period average.
# Exit: Opposite Donchian breakout or ATR trailing stop (2.0x ATR from extreme).
# Uses daily Donchian for structure, 6h EMA for pullback entry, volume for confirmation, ATR for dynamic stops.
# Target: 12-37 trades/year (50-150 total over 4 years) on BTC/ETH/SOL.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_donchian_volume_pullback_v1"
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
    
    # Get 1d data for Donchian channel (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    # Calculate daily Donchian(20) channels
    high_1d = pd.Series(df_1d['high'].values)
    low_1d = pd.Series(df_1d['low'].values)
    donchian_high = high_1d.rolling(window=20, min_periods=20).max().values
    donchian_low = low_1d.rolling(window=20, min_periods=20).min().values
    
    # Align HTF Donchian levels to 6h timeframe (wait for completed 1d bar)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Calculate 6h EMA(20) for pullback entry
    close_s = pd.Series(close)
    ema20_6h = close_s.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    long_high = 0.0   # highest high since long entry
    short_low = 0.0   # lowest low since short entry
    long_triggered = False  # flag to wait for pullback after breakout
    short_triggered = False  # flag to wait for pullback after breakout
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(volume_ma[i]) or np.isnan(atr[i]) or np.isnan(close[i]) or np.isnan(high[i]) or np.isnan(low[i]) or
            np.isnan(open_price[i]) or np.isnan(volume[i]) or np.isnan(ema20_6h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Update highest high since entry
            long_high = max(long_high, high[i])
            # ATR trailing stop: exit if price drops 2.0*ATR from high
            if long_high > 0 and close[i] < long_high - 2.0 * atr[i]:
                position = 0
                long_high = 0.0
                long_triggered = False
                signals[i] = 0.0
            # Exit: Price breaks below daily Donchian low
            elif close[i] < donchian_low_aligned[i]:
                position = 0
                long_high = 0.0
                long_triggered = False
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
                short_triggered = False
                signals[i] = 0.0
            # Exit: Price breaks above daily Donchian high
            elif close[i] > donchian_high_aligned[i]:
                position = 0
                short_low = 0.0
                short_triggered = False
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Check for breakout and set trigger flags
            bullish_breakout = (close[i] > donchian_high_aligned[i]) and volume_confirmed
            bearish_breakout = (close[i] < donchian_low_aligned[i]) and volume_confirmed
            
            if bullish_breakout:
                long_triggered = True
                short_triggered = False
            elif bearish_breakout:
                short_triggered = True
                long_triggered = False
            
            # Long entry: after bullish breakout, price pulls back to EMA20 or above
            if long_triggered and close[i] >= ema20_6h[i]:
                position = 1
                long_high = high[i]
                long_triggered = False
                signals[i] = 0.25
            # Short entry: after bearish breakout, price pulls back to EMA20 or below
            elif short_triggered and close[i] <= ema20_6h[i]:
                position = -1
                short_low = low[i]
                short_triggered = False
                signals[i] = -0.25
    
    return signals