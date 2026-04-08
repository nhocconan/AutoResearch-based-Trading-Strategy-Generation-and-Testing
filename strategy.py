#!/usr/bin/env python3
"""
1d Weekly Donchian Breakout with Volume Confirmation and Trend Filter
Hypothesis: Weekly Donchian channels identify key support/resistance levels.
Breakouts above weekly high or below weekly low with volume confirmation and
trend alignment (using daily EMA) capture strong moves in both bull and bear markets.
Volatility filter (ATR) avoids choppy markets. Target: 10-25 trades/year on 1d.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_donchian_breakout_volume_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly OHLC for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Donchian channels (20-period)
    lookback = 20
    upper_20 = np.full(len(high_1w), np.nan)
    lower_20 = np.full(len(low_1w), np.nan)
    
    for i in range(lookback, len(high_1w)):
        upper_20[i] = np.max(high_1w[i-lookback:i])
        lower_20[i] = np.min(low_1w[i-lookback:i])
    
    # Align Donchian levels to daily timeframe (shifted by 1 for completed weekly bars only)
    upper_20_aligned = align_htf_to_ltf(prices, df_1w, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1w, lower_20)
    
    # Daily EMA(50) for trend filter
    ema_50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # ATR(14) for volatility filter
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    true_range[0] = high_low[0]  # First value
    atr = pd.Series(true_range).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter (>1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    # Volatility filter: avoid extremely high volatility (ATR > 2x its 50-period average)
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    vol_regime_filter = atr < (atr_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(ema_50[i]) or np.isnan(vol_filter[i]) or 
            np.isnan(vol_regime_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below weekly Donchian lower OR trend turns bearish
            if (close[i] <= lower_20_aligned[i] or 
                close[i] < ema_50[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above weekly Donchian upper OR trend turns bullish
            if (close[i] >= upper_20_aligned[i] or 
                close[i] > ema_50[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price breaks above weekly Donchian upper with uptrend and volume
            if (close[i] >= upper_20_aligned[i] and 
                close[i] > ema_50[i] and 
                vol_filter[i] and 
                vol_regime_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below weekly Donchian lower with downtrend and volume
            elif (close[i] <= lower_20_aligned[i] and 
                  close[i] < ema_50[i] and 
                  vol_filter[i] and 
                  vol_regime_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals