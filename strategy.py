#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Donchian(20) breakout with 1d volume confirmation and ATR-based trend filter
    # Designed for low trade frequency (12-37/year) to minimize fee drag
    # Works in bull/bear markets by capturing breakouts with volume confirmation
    # Uses 1d ATR to filter for trending markets only (avoid chop)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values if 'volume' in prices.columns else np.ones(len(prices))
    
    # Get 1d data for HTF indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values if 'volume' in df_1d.columns else np.ones(len(df_1d))
    
    # Calculate 12h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d ATR(14) for trend filter and volatility
    tr1 = np.maximum(high_1d[1:], close_1d[:-1]) - np.minimum(low_1d[1:], close_1d[:-1])
    tr1 = np.concatenate([[np.nan], tr1])
    atr_1d = pd.Series(tr1).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 1d EMA50 for trend direction
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d volume average (20-period)
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF indicators to 12h primary timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or
            np.isnan(atr_1d_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or
            np.isnan(vol_avg_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.8x 20-period average
        volume_confirmed = volume_1d[i] > 1.8 * vol_avg_20_1d_aligned[i]
        
        # Trend filter: only trade when ATR is expanding (trending market)
        # Compare current ATR to ATR 5 periods ago
        if i >= 5:
            atr_expanding = atr_1d_aligned[i] > atr_1d_aligned[i-5]
        else:
            atr_expanding = False
        
        # Donchian breakout conditions
        breakout_long = close[i] > donchian_high[i-1]  # Break above upper band
        breakout_short = close[i] < donchian_low[i-1]  # Break below lower band
        
        # Entry conditions: breakout + volume + expanding ATR (trend)
        enter_long = breakout_long and volume_confirmed and atr_expanding
        enter_short = breakout_short and volume_confirmed and atr_expanding
        
        # Exit conditions: opposite Donchian breakout or ATR contraction
        exit_long = position == 1 and (close[i] < donchian_low[i-1] or not atr_expanding)
        exit_short = position == -1 and (close[i] > donchian_high[i-1] or not atr_expanding)
        
        # Execute signals
        if enter_long and position != 1:
            position = 1
            signals[i] = position_size
        elif enter_short and position != -1:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_donchian_breakout_volume_atr_filter_v1"
timeframe = "12h"
leverage = 1.0