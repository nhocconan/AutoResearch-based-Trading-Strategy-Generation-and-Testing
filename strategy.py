#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1w EMA50 trend filter and volume spike confirmation.
# Enter long when price breaks above R3 with volume > 2x 24-bar average and 1w EMA50 trending up.
# Enter short when price breaks below S3 with volume > 2x 24-bar average and 1w EMA50 trending down.
# Exit when price retouches the 1d EMA34 or after 8 bars (max hold).
# Uses discrete position sizing (0.30) to limit drawdown. Target: 50-150 total trades over 4 years (12-37/year).
# Camarilla levels provide intraday support/resistance; 1w EMA50 ensures higher timeframe alignment;
# volume spike confirms institutional interest. Works in both bull (strong breakouts) and bear (strong breakdowns).

name = "12h_Camarilla_R3S3_Breakout_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels and EMA34 exit
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for exit condition
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    start_idx = 50  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Calculate Camarilla levels from previous 1d bar
        # Need to get the index of the last completed 1d bar
        # Since we're on 12h timeframe, we look back 2 bars for previous day
        if i >= 2:
            prev_day_idx = i - 2
            if prev_day_idx < len(prices):
                # Get high, low, close from previous day (2 bars ago on 12h TF)
                ph = high[prev_day_idx]
                pl = low[prev_day_idx]
                pc = close[prev_day_idx]
                
                # Calculate Camarilla levels
                range_ = ph - pl
                if range_ > 0:
                    r3 = pc + (range_ * 1.1 / 4)
                    s3 = pc - (range_ * 1.1 / 4)
                else:
                    r3 = pc
                    s3 = pc
            else:
                # Not enough history, skip
                signals[i] = 0.0
                continue
        else:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(r3) or np.isnan(s3)):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: >2x 24-bar average volume (2 days on 12h TF)
        if i >= 24:
            volume_ma_24 = np.mean(volume[i-24:i])
            volume_confirm = volume[i] > 2.0 * volume_ma_24
        else:
            volume_confirm = False
        
        # 1w EMA50 trend: slope over 3 periods
        if i >= 3:
            ema_slope = (ema_50_1w_aligned[i] - ema_50_1w_aligned[i-3]) / 3
            ema_trend_up = ema_slope > 0
            ema_trend_down = ema_slope < 0
        else:
            ema_trend_up = False
            ema_trend_down = False
        
        # Price action
        price = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above R3 with volume confirm and 1w EMA50 up
            if price > r3 and volume_confirm and ema_trend_up:
                signals[i] = 0.30
                position = 1
                bars_since_entry = 0
            # Short entry: price breaks below S3 with volume confirm and 1w EMA50 down
            elif price < s3 and volume_confirm and ema_trend_down:
                signals[i] = -0.30
                position = -1
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - hold or exit
            bars_since_entry += 1
            # Exit conditions: price retouches 1d EMA34 or max hold (8 bars = 4 days)
            if price <= ema_34_1d_aligned[i] or bars_since_entry >= 8:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.30
        elif position == -1:  # Short - hold or exit
            bars_since_entry += 1
            # Exit conditions: price retouches 1d EMA34 or max hold (8 bars = 4 days)
            if price >= ema_34_1d_aligned[i] or bars_since_entry >= 8:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.30
    
    return signals