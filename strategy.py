#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout with 1w trend filter and volume confirmation
# Long when price breaks above Donchian(20) upper band AND 1w EMA(21) is rising AND volume > 1.5x average
# Short when price breaks below Donchian(20) lower band AND 1w EMA(21) is falling AND volume > 1.5x average
# Exit when price crosses back through Donchian middle (mean reversion) or opposite breakout occurs
# Donchian channels capture breakouts; 1w EMA ensures higher timeframe trend alignment; volume confirms institutional interest
# Designed to work in both bull and bear markets by following the dominant trend on 1w timeframe
# Target: 30-100 total trades over 4 years (7-25/year) to balance opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Donchian channels on 1d (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_mid = (highest_high + lowest_low) / 2
    
    # Calculate EMA on 1w (21-period) for trend filter
    ema_21 = pd.Series(df_1w['close']).ewm(span=21, adjust=False, min_periods=21).mean()
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(donchian_mid[i])):
            signals[i] = 0.0
            continue
        
        # Get EMA values aligned to 1d timeframe
        ema_21_aligned = align_htf_to_ltf(prices, df_1w, ema_21.values)
        ema_val = ema_21_aligned[i]
        ema_prev = ema_21_aligned[i-1]
        
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        if position == 0:
            # Long setup: price breaks above Donchian upper AND 1w EMA rising AND volume confirmation
            if (high_val > highest_high[i] and ema_val > ema_prev and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: price breaks below Donchian lower AND 1w EMA falling AND volume confirmation
            elif (low_val < lowest_low[i] and ema_val < ema_prev and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below Donchian middle OR opposite breakout
            if (close_val < donchian_mid[i] or 
                low_val < lowest_low[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above Donchian middle OR opposite breakout
            if (close_val > donchian_mid[i] or 
                high_val > highest_high[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_Donchian_1wEMA_Volume"
timeframe = "1d"
leverage = 1.0