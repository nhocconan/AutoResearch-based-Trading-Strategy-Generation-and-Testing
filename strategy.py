#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike.
# Enter long when price breaks above Camarilla R3 level with 1d EMA34 uptrend and volume > 2x 20-bar average.
# Enter short when price breaks below Camarilla S3 level with 1d EMA34 downtrend and volume > 2x 20-bar average.
# Exit when price retraces to the Camarilla pivot point (PP).
# Uses discrete position sizing (0.25) to limit drawdown and reduce fee churn.
# Target: 50-120 total trades over 4 years (12-30/year).
# Camarilla levels provide precise intraday support/resistance; 1d EMA34 ensures higher timeframe alignment;
# volume spike filters weak breakouts. Works in both bull (strong breakouts) and bear (strong breakdowns).

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA34 to 4h
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate Camarilla levels from previous 1d bar (use prior completed day)
    # Typical price = (high + low + close) / 3
    typical_price = (high + low + close) / 3
    # Use prior day's typical price (shifted by 1) to avoid look-ahead
    typical_price_prior = pd.Series(typical_price).shift(1)
    # Rolling window of 1 day (since we're on 4h timeframe, need to get prior day's values)
    # For 4h data, we need to look back 6 bars to get prior day's typical price (approx)
    # Better: use the 1d data directly for Camarilla calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Prior day's typical price (yesterday's)
    typical_price_1d = (high_1d + low_1d + close_1d) / 3
    typical_price_1d_prior = pd.Series(typical_price_1d).shift(1)  # yesterday's typical price
    
    # Calculate Camarilla levels for each 1d bar
    # R4 = PP + ((H-L) * 1.1/2)
    # R3 = PP + ((H-L) * 1.1/4)
    # S3 = PP - ((H-L) * 1.1/4)
    # PP = (H+L+C)/3
    range_1d = high_1d - low_1d
    camarilla_pp = typical_price_1d
    camarilla_r3 = camarilla_pp + (range_1d * 1.1 / 4)
    camarilla_s3 = camarilla_pp - (range_1d * 1.1 / 4)
    
    # Align Camarilla levels to 4h (using prior completed day's levels)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp, additional_delay_bars=1)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3, additional_delay_bars=1)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3, additional_delay_bars=1)
    
    # Volume confirmation: >2x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Ensure sufficient history for EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(volume_ma_20[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_pp_aligned[i])):
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
            # Long entry: price > Camarilla R3, EMA34 up, volume confirm
            if price > camarilla_r3_aligned[i] and ema_trend_up and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: price < Camarilla S3, EMA34 down, volume confirm
            elif price < camarilla_s3_aligned[i] and ema_trend_down and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - hold or exit at Camarilla PP
            if price <= camarilla_pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - hold or exit at Camarilla PP
            if price >= camarilla_pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals