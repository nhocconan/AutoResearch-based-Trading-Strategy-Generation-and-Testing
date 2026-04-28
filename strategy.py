#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
# Enter long when price breaks above R3 level with volume > 2x 20-bar average and 1d EMA34 trending up.
# Enter short when price breaks below S3 level with volume > 2x 20-bar average and 1d EMA34 trending down.
# Exit when price returns to the Camarilla pivot level (PP) or opposite S1/R1 level.
# Uses discrete position sizing (0.30) to limit drawdown. Target: 60-120 total trades over 4 years (15-30/year).
# Camarilla levels provide precise intraday support/resistance; 1d EMA34 ensures higher timeframe alignment;
# volume spike filters breakout strength. Works in both bull (strong breakouts) and bear (strong breakdowns).

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA34 to 12h
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate 1d EMA34 trend (slope over 3 periods)
    ema_34_slope = pd.Series(ema_34_aligned).diff(3) / 3
    ema_trend_up = ema_34_slope > 0
    ema_trend_down = ema_34_slope < 0
    
    # Calculate Camarilla levels from previous 1d bar
    # Camarilla: PP = (H+L+C)/3, R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), etc.
    # We use R3, S3, PP, S1, R1 for entries/exits
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # First bar has no previous day
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate Camarilla levels
    pp = (prev_high + prev_low + prev_close) / 3
    r3 = prev_close + ((prev_high - prev_low) * 1.1 / 4)
    s3 = prev_close - ((prev_high - prev_low) * 1.1 / 4)
    r1 = prev_close + ((prev_high - prev_low) * 1.1 / 12)
    s1 = prev_close - ((prev_high - prev_low) * 1.1 / 12)
    
    # Align Camarilla levels to 12h
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume confirmation: >2x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(pp_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        # 1d EMA34 trend
        trend_up = ema_trend_up.iloc[i] if hasattr(ema_trend_up, 'iloc') else ema_trend_up[i]
        trend_down = ema_trend_down.iloc[i] if hasattr(ema_trend_down, 'iloc') else ema_trend_down[i]
        
        # Price action
        price = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: price > R3, volume spike, EMA34 up
            if (price > r3_aligned[i] and vol_spike and trend_up):
                signals[i] = 0.30
                position = 1
            # Short entry: price < S3, volume spike, EMA34 down
            elif (price < s3_aligned[i] and vol_spike and trend_down):
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - hold or exit
            # Exit when price returns to PP or drops below S1
            if (price <= pp_aligned[i] or price < s1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:  # Short - hold or exit
            # Exit when price returns to PP or rises above R1
            if (price >= pp_aligned[i] or price > r1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals