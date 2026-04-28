#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R extremes with 1d EMA34 trend filter and volume confirmation.
# Enter long when Williams %R crosses above -80 (oversold), 1d EMA34 trending up, and volume > 1.8x 20-bar average.
# Enter short when Williams %R crosses below -20 (overbought), 1d EMA34 trending down, and volume > 1.8x 20-bar average.
# Exit when Williams %R returns to the opposite extreme zone (long exit at -20, short exit at -80).
# Uses discrete position sizing (0.25) to limit drawdown. Target: 80-160 total trades over 4 years (20-40/year).
# Williams %R identifies momentum extremes that often precede reversals in ranging markets, while 1d EMA34
# ensures alignment with higher timeframe trend. Volume confirmation filters weak signals. Works in both bull
# (mean reversion from oversold) and bear (mean reversion from overbought) regimes.

name = "4h_WilliamsR_Extremes_1dEMA34_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R calculation and EMA34 trend
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d Williams %R (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 1d EMA34 for trend filter
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align Williams %R and EMA34 to 4h
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume confirmation: >1.8x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.8 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
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
        
        # Williams %R levels
        wr = williams_r_aligned[i]
        
        # Extreme zones
        oversold = wr < -80
        overbought = wr > -20
        
        # Exit conditions: return to opposite extreme
        exit_long = wr > -20  # Exit long when no longer oversold
        exit_short = wr < -80  # Exit short when no longer overbought
        
        # Entry conditions: cross into extreme with trend and volume
        # Long: Williams %R crosses above -80 from below
        long_entry = (wr > -80) and (i > start_idx) and (williams_r_aligned[i-1] <= -80)
        # Short: Williams %R crosses below -20 from above
        short_entry = (wr < -20) and (i > start_idx) and (williams_r_aligned[i-1] >= -20)
        
        # Handle entries and exits
        if long_entry and ema_trend_up and vol_confirm and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and ema_trend_down and vol_confirm and position >= 0:
            signals[i] = -0.25
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
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