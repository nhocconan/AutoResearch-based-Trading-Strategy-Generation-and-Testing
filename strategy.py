# 103293
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data once for HTF context (as specified in experiment)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h indicators
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # 12h EMA(50) for trend
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 4h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 4h Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h ATR(14) for volatility
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter: above average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate hourly filter (8-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(atr_14[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 8-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            # Outside session: flatten position
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 12h EMA50
        trend_up = close[i] > ema_50_12h_aligned[i]
        trend_down = close[i] < ema_50_12h_aligned[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > donchian_high[i-1]  # Break above previous high
        breakout_down = close[i] < donchian_low[i-1]  # Break below previous low
        
        # Volume filter: above average volume
        vol_filter = volume[i] > vol_ma[i]
        
        # Entry conditions with strict criteria
        long_entry = trend_up and breakout_up and vol_filter
        short_entry = trend_down and breakout_down and vol_filter
        
        # Exit conditions: ATR-based trailing stop
        if position == 1:  # Long position
            # Calculate highest high since entry (simplified: use recent high)
            recent_high = np.max(high[max(0, i-20):i+1])
            long_exit = close[i] < recent_high - 2.5 * atr_14[i]
        elif position == -1:  # Short position
            # Calculate lowest low since entry (simplified: use recent low)
            recent_low = np.min(low[max(0, i-20):i+1])
            short_exit = close[i] > recent_low + 2.5 * atr_14[i]
        else:
            long_exit = False
            short_exit = False
        
        # Execute trades
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (long_exit and position == 1) or (short_exit and position == -1):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Donchian_Breakout_12hEMA50_Volume_Filter"
timeframe = "4h"
leverage = 1.0