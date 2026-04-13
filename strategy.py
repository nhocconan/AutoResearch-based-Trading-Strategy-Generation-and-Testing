# 1h_4h_1d_camarilla_breakout_v1
# Hypothesis: Use 1h timeframe for entry timing with 4h/1d multi-timeframe confirmation.
# 4h Camarilla levels (H4/L4) provide directional bias, 1d volume confirms strength,
# and 1h provides precise entry timing. Designed to work in both bull and bear markets
# by requiring trend alignment and volume confirmation to avoid false breakouts.
# Target: 60-150 total trades over 4 years (15-37/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume confirmation (using previous day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (vol_ma_20 * 1.5)  # Volume > 1.5x 20-day average
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike.astype(float))
    
    # Get 4h data for Camarilla levels (based on previous day's OHLC)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate 4h Camarilla levels using previous day's OHLC
    df_1d_ohlc = get_htf_data(prices, '1d')
    if len(df_1d_ohlc) < 2:
        return np.zeros(n)
    
    prev_close = df_1d_ohlc['close'].shift(1).values
    prev_high = df_1d_ohlc['high'].shift(1).values
    prev_low = df_1d_ohlc['low'].shift(1).values
    
    # Camarilla levels: H4 = close + 1.1*(high-low)*1.1/2, L4 = close - 1.1*(high-low)*1.1/2
    camarilla_h4 = prev_close + 1.1 * (prev_high - prev_low) * 1.1 / 2
    camarilla_l4 = prev_close - 1.1 * (prev_high - prev_low) * 1.1 / 2
    
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d_ohlc, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d_ohlc, camarilla_l4)
    
    # Session filter: 08:00-20:00 UTC (active trading hours)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.20  # 20% of capital
    
    for i in range(20, n):
        # Skip if data not ready or outside session
        if (np.isnan(camarilla_h4_aligned[i]) or 
            np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(vol_spike_aligned[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Entry conditions: Camarilla breakout + volume spike
        breakout_long = close[i] > camarilla_h4_aligned[i]
        breakout_short = close[i] < camarilla_l4_aligned[i]
        vol_confirm = vol_spike_aligned[i] > 0.5  # True if volume spike
        
        long_entry = breakout_long and vol_confirm
        short_entry = breakout_short and vol_confirm
        
        # Exit when price returns to Camarilla center (previous day's close)
        camarilla_close_aligned = align_htf_to_ltf(prices, df_1d_ohlc, prev_close)
        exit_long = position == 1 and close[i] < camarilla_close_aligned[i]
        exit_short = position == -1 and close[i] > camarilla_close_aligned[i]
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
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

name = "1h_4h_1d_camarilla_breakout_v1"
timeframe = "1h"
leverage = 1.0