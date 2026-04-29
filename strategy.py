#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation
# Long when price breaks above 20-day high AND price > 1w EMA34 AND volume > 2.0x 20-bar avg
# Short when price breaks below 20-day low AND price < 1w EMA34 AND volume > 2.0x 20-bar avg
# Exit when price crosses opposite Donchian level (20-day low for longs, 20-day high for shorts)
# Uses discrete position sizing (0.25) to minimize fee churn while capturing moves.
# Target: 30-100 total trades over 4 years (7-25/year) on 1d.
# Donchian channels capture strong trends, 1w EMA34 filters counter-trend moves, volume spike ensures institutional participation.
# Works in bull markets (trend continuation via breakouts) and bear markets (mean reversion within trend via exits).

name = "1d_Donchian20_1wEMA34_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate EMA(34) on 1w data
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Align EMA34 to 1d timeframe
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Donchian channels (20-period) on 1d data
    # We need the completed prior bar's high/low to avoid look-ahead
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20) + 1  # EMA34 warmup + Donchian warmup + 1 for shift
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_close = close[i]
        ema_34 = ema_34_1w_aligned[i]
        
        # Donchian levels
        donch_high = donchian_high[i]
        donch_low = donchian_low[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price crosses below 20-day low (mean reversion)
            if curr_close < donch_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above 20-day high (mean reversion)
            if curr_close > donch_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above 20-day high AND price > 1w EMA34 AND volume confirmation
            if curr_close > donch_high and curr_close > ema_34 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below 20-day low AND price < 1w EMA34 AND volume confirmation
            elif curr_close < donch_low and curr_close < ema_34 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals