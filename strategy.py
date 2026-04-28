#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R4/S4 breakout with 1d EMA34 trend filter and volume spike confirmation.
# Enter long when price breaks above Camarilla R4 level with 1d EMA34 uptrend and volume > 2.5x 24-bar average.
# Enter short when price breaks below Camarilla S4 level with 1d EMA34 downtrend and volume > 2.5x 24-bar average.
# Exit when price retraces to the Camarilla H4/L4 levels respectively.
# Uses discrete position sizing (0.30) to balance return and drawdown.
# Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe.
# Camarilla R4/S4 levels represent stronger breakout points than R3/S3; 1d EMA34 ensures daily trend alignment;
# volume spike filters weak breakouts. Designed to work in both bull (strong breakouts) and bear (strong breakdowns)
# regimes by requiring trend and volume confirmation.

name = "12h_Camarilla_R4S4_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA34 to 12h
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Previous day's high, low, close for Camarilla
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Align to 12h
    prev_high_aligned = align_htf_to_ltf(prices, df_1d, prev_high)
    prev_low_aligned = align_htf_to_ltf(prices, df_1d, prev_low)
    prev_close_aligned = align_htf_to_ltf(prices, df_1d, prev_close)
    
    # Camarilla levels (R4/S4 and H4/L4)
    range_ = prev_high_aligned - prev_low_aligned
    R4 = prev_close_aligned + range_ * 1.1 / 2
    S4 = prev_close_aligned - range_ * 1.1 / 2
    H4 = prev_close_aligned + range_ * 1.1 / 4
    L4 = prev_close_aligned - range_ * 1.1 / 4
    
    # Volume confirmation: >2.5x 24-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_24 = volume_series.rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > 2.5 * volume_ma_24
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 24  # Ensure sufficient history for volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(volume_ma_24[i]) or 
            np.isnan(prev_high_aligned[i]) or np.isnan(prev_low_aligned[i]) or 
            np.isnan(prev_close_aligned[i]) or np.isnan(R4[i]) or np.isnan(S4[i]) or
            np.isnan(H4[i]) or np.isnan(L4[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # 1d EMA34 trend: slope over 3 periods
        if i >= 3:
            ema_slope = (ema_34_aligned[i] - ema_34_aligned[i-3]) / 3
            ema_trend_up = ema_slope > 0
            ema_trend_down = ema_slope < 0
        else:
            ema_trend_up = False
            ema_trend_down = False
        
        price = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: price > R4, EMA34 up, volume confirm
            if price > R4[i] and ema_trend_up and vol_confirm:
                signals[i] = 0.30
                position = 1
            # Short entry: price < S4, EMA34 down, volume confirm
            elif price < S4[i] and ema_trend_down and vol_confirm:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - hold or exit at H4
            if price <= H4[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:  # Short - hold or exit at L4
            if price >= L4[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals