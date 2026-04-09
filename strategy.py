#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Donchian channel breakout with ATR-based volatility filter and volume confirmation
# 1w Donchian channels provide major structural support/resistance levels that work across market regimes
# Volume confirmation ensures breakouts have conviction
# ATR filter avoids trading in excessively choppy conditions
# Designed for low trade frequency (<25/year) to minimize fee drag while capturing major trend moves
# Works in bull markets (breakouts continuation) and bear markets (breakdowns continuation)

name = "1d_1w_donchian_volume_atr_v1"
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
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Donchian channels (20-period)
    # Upper channel = highest high over 20 periods
    # Lower channel = lowest low over 20 periods
    high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1d timeframe
    donchian_high = align_htf_to_ltf(prices, df_1w, high_20)
    donchian_low = align_htf_to_ltf(prices, df_1w, low_20)
    
    # Calculate ATR for volatility filter (using 1d data)
    # ATR(14) = average true range over 14 periods
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x average 1d volume
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        # ATR filter: only trade when volatility is reasonable (not too high, not too low)
        # Avoid extremely high volatility (panic conditions) and extremely low volatility (chop)
        atr_ma_50 = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
        if not np.isnan(atr_ma_50[i]) and atr_ma_50[i] > 0:
            atr_ratio = atr[i] / atr_ma_50[i]
            volatility_filter = (atr_ratio > 0.5) & (atr_ratio < 2.0)
        else:
            volatility_filter = True
        
        if position == 1:  # Long position
            # Exit when price closes below Donchian lower channel (trend reversal)
            if close[i] < donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when price closes above Donchian upper channel (trend reversal)
            if close[i] > donchian_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Breakout trading with volume and volatility filters
            # Long on Donchian upper channel breakout
            # Short on Donchian lower channel breakdown
            if volume_confirmed and volatility_filter:
                if close[i] > donchian_high[i]:
                    position = 1
                    signals[i] = 0.25
                elif close[i] < donchian_low[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals