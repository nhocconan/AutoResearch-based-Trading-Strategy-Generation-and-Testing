#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R3/S3 breakout with 1w EMA50 trend filter and volume confirmation.
# Enter long when price breaks above Camarilla R3 level, 1w EMA50 trending up, and volume > 2.0x 20-bar average.
# Enter short when price breaks below Camarilla S3 level, 1w EMA50 trending down, and volume > 2.0x 20-bar average.
# Exit when price returns to Camarilla H3/L3 levels (mean of H3 and L3).
# Uses discrete position sizing (0.25) to limit drawdown. Target: 30-100 total trades over 4 years (7-25/year).
# Camarilla levels provide intraday structure; 1w EMA50 ensures alignment with higher timeframe trend;
# volume confirmation filters weak breakouts. Works in both bull (breakouts) and bear (breakdowns).

name = "1d_Camarilla_R3S3_Breakout_1wEMA50_Trend_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA50 to 1d
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Calculate Camarilla levels on daily data
    # Based on previous day's high, low, close
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    
    # First day has no previous data
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    # Camarilla levels calculation
    # R3 = Close + (High - Low) * 1.1/2
    # S3 = Close - (High - Low) * 1.1/2
    # H3 = Close + (High - Low) * 1.1/4
    # L3 = Close - (High - Low) * 1.1/4
    range_hl = prev_high - prev_low
    camarilla_r3 = prev_close + range_hl * 1.1 / 2
    camarilla_s3 = prev_close - range_hl * 1.1 / 2
    camarilla_h3 = prev_close + range_hl * 1.1 / 4
    camarilla_l3 = prev_close - range_hl * 1.1 / 4
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # 1w EMA50 trend: slope over 3 periods
        if i >= 3:
            ema_slope = (ema_50_aligned[i] - ema_50_aligned[i-3]) / 3
            ema_trend_up = ema_slope > 0
            ema_trend_down = ema_slope < 0
        else:
            ema_trend_up = False
            ema_trend_down = False
        
        # Price action
        price = close[i]
        
        # Breakout conditions
        breakout_long = price > camarilla_r3[i]
        breakout_short = price < camarilla_s3[i]
        
        # Exit conditions: return to H3/L3 levels
        exit_long = price < camarilla_h3[i]
        exit_short = price > camarilla_l3[i]
        
        # Handle entries and exits
        if breakout_long and ema_trend_up and vol_confirm and position <= 0:
            signals[i] = 0.25
            position = 1
        elif breakout_short and ema_trend_down and vol_confirm and position >= 0:
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