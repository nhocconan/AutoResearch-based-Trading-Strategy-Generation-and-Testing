#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h EMA34 trend filter and volume confirmation.
# Enter long when price breaks above Camarilla R3 level, 4h EMA34 trending up, and volume > 2.0x 20-bar average.
# Enter short when price breaks below Camarilla S3 level, 4h EMA34 trending down, and volume > 2.0x 20-bar average.
# Exit when price returns to Camarilla Pivot level (midpoint).
# Uses discrete position sizing (0.20) to limit drawdown in bear markets like 2022.
# Target: 60-150 total trades over 4 years (15-37/year) to balance opportunity and fee drag.
# Camarilla levels identify intraday support/resistance; 4h EMA34 ensures alignment with higher timeframe trend;
# high volume threshold confirms institutional participation. Works in both bull (breakouts) and bear (breakdowns).

name = "1h_Camarilla_R3S3_Breakout_4hEMA34_Trend_VolumeSpike_v2"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation (based on previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    # Camarilla: based on previous day's high, low, close
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift by 1 to use previous day's data (no look-ahead)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan  # First value has no previous day
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate Camarilla levels
    # R4 = Close + (High - Low) * 1.5/2
    # R3 = Close + (High - Low) * 1.25/2
    # R2 = Close + (High - Low) * 1.1/2
    # R1 = Close + (High - Low) * 1.05/2
    # PP = (High + Low + Close) / 3
    # S1 = Close - (High - Low) * 1.05/2
    # S2 = Close - (High - Low) * 1.1/2
    # S3 = Close - (High - Low) * 1.25/2
    # S4 = Close - (High - Low) * 1.5/2
    
    diff = prev_high - prev_low
    camarilla_pp = (prev_high + prev_low + prev_close) / 3
    camarilla_r3 = prev_close + diff * 1.25 / 2
    camarilla_s3 = prev_close - diff * 1.25 / 2
    camarilla_r4 = prev_close + diff * 1.5 / 2
    camarilla_s4 = prev_close - diff * 1.5 / 2
    camarilla_s1 = prev_close - diff * 1.05 / 2
    camarilla_r1 = prev_close + diff * 1.05 / 2
    
    # Align Camarilla levels to 1h
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    
    # Get 4h data for EMA34 trend filter
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 34:
        return np.zeros(n)
    
    # Calculate 4h EMA34
    close_4h = df_4h['close'].values
    ema_34 = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA34 to 1h
    ema_34_aligned = align_htf_to_ltf(prices, df_4h, ema_34)
    
    # Volume confirmation: >2.0x 20-bar average volume (strict to reduce trades)
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    # Session filter: 08-20 UTC (reduce noise trades)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # Session filter
        session_ok = in_session[i]
        
        # 4h EMA34 trend: slope over 3 periods
        if i >= 3:
            ema_slope = (ema_34_aligned[i] - ema_34_aligned[i-3]) / 3
            ema_trend_up = ema_slope > 0
            ema_trend_down = ema_slope < 0
        else:
            ema_trend_up = False
            ema_trend_down = False
        
        # Price action
        price = close[i]
        
        # Breakout conditions
        breakout_long = price > camarilla_r3_aligned[i]
        breakout_short = price < camarilla_s3_aligned[i]
        
        # Exit conditions: return to pivot level
        exit_long = price < camarilla_pp_aligned[i]
        exit_short = price > camarilla_pp_aligned[i]
        
        # Handle entries and exits
        if breakout_long and ema_trend_up and vol_confirm and session_ok and position <= 0:
            signals[i] = 0.20
            position = 1
        elif breakout_short and ema_trend_down and vol_confirm and session_ok and position >= 0:
            signals[i] = -0.20
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
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals