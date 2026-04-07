#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1d Donchian(20) breakout with weekly trend filter and volume confirmation
# Hypothesis: Daily breakouts filtered by weekly trend capture major moves while avoiding counter-trend trades.
# Volume confirmation ensures breakout authenticity. Works in bull via breakouts, in bear via trend-filtered mean reversion.
# Target: 10-25 trades/year to minimize fee drag.
name = "1d_donchian20_1w_trend_volume_v10"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly 50-period EMA for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate daily 20-period volume moving average
    vol_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR(14) for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian channels (20-period high/low)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_20[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 20-period average
        vol_confirm = volume[i] > vol_20[i]
        
        # Volatility filter: only trade when ATR is above its 50-period average (avoid low volatility chop)
        atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
        vol_filter = atr[i] > atr_ma[i] if not np.isnan(atr_ma[i]) else True
        
        if position == 1:  # Long position
            # Exit: price touches lower band OR trend turns bearish OR volatility drops
            if close[i] <= lowest_low[i] or close[i] < ema_50_1w_aligned[i] or not vol_filter:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price touches upper band OR trend turns bullish OR volatility drops
            if close[i] >= highest_high[i] or close[i] > ema_50_1w_aligned[i] or not vol_filter:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: price breaks above upper band + weekly uptrend + volume confirmation + volatility filter
            if close[i] > highest_high[i] and close[i] > ema_50_1w_aligned[i] and vol_confirm and vol_filter:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below lower band + weekly downtrend + volume confirmation + volatility filter
            elif close[i] < lowest_low[i] and close[i] < ema_50_1w_aligned[i] and vol_confirm and vol_filter:
                position = -1
                signals[i] = -0.25
    
    return signals