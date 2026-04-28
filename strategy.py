#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extremes with 1d EMA34 trend filter and volume confirmation.
# Enter long when Williams %R(14) crosses above -80 (oversold), 1d EMA34 trending up, and volume > 1.8x 20-bar average.
# Enter short when Williams %R(14) crosses below -20 (overbought), 1d EMA34 trending down, and volume > 1.8x 20-bar average.
# Exit when Williams %R crosses back through -50 (mean reversion) or opposite extreme is reached.
# Williams %R identifies exhaustion points in ranging markets; EMA34 filters for 1d trend alignment to avoid counter-trend trades;
# Volume confirmation ensures participation. Designed for 6h timeframe to capture mean reversion in both bull and bear regimes.
# Target: 80-160 total trades over 4 years (20-40/year) to balance opportunity and fee drag.

name = "6h_WilliamsR_Extremes_1dEMA34_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams %R(14) on 6h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
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
        if (np.isnan(williams_r[i]) or np.isnan(ema_34_aligned[i]) or 
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
        
        # Williams %R conditions
        wr = williams_r[i]
        wr_prev = williams_r[i-1]
        
        # Long: WR crosses above -80 from below (oversold bounce)
        long_entry = (wr_prev <= -80) and (wr > -80) and ema_trend_up and vol_confirm
        # Short: WR crosses below -20 from above (overbought rejection)
        short_entry = (wr_prev >= -20) and (wr < -20) and ema_trend_down and vol_confirm
        
        # Exit conditions: WR crosses -50 (mean reversion) or reaches opposite extreme
        exit_long = (wr_prev > -50) and (wr <= -50)  # WR crosses below -50
        exit_short = (wr_prev < -50) and (wr >= -50)  # WR crosses above -50
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
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