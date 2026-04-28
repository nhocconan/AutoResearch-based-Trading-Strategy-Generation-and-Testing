#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Camarilla pivot levels calculated from 1d OHLC: R3 = close + 1.1*(high-low)*1.1/4, S3 = close - 1.1*(high-low)*1.1/4
# Long when price breaks above R3 with volume > 2.0x 20-bar average and price > 1d EMA34
# Short when price breaks below S3 with volume > 2.0x 20-bar average and price < 1d EMA34
# Exit on opposite Camarilla level touch (R3 for longs, S3 for shorts) or volume drying up
# Uses 12h timeframe targeting 12-37 trades/year (~50-150 total over 4 years) to minimize fee drag.
# Camarilla levels provide precise support/resistance; EMA34 filter ensures higher-timeframe alignment.

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike_v1"
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
    
    # Get 1d data for EMA trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Camarilla levels (R3, S3, R4, S4)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla width
    width_1d = high_1d - low_1d
    # Camarilla multipliers
    camarilla_multiplier = 1.1 / 4  # 0.275
    
    # R3 = close + 1.1*(high-low)*1.1/4 = close + width * 1.1 * 1.1/4
    # S3 = close - 1.1*(high-low)*1.1/4 = close - width * 1.1 * 1.1/4
    r3_1d = close_1d + width_1d * camarilla_multiplier * 1.1
    s3_1d = close_1d - width_1d * camarilla_multiplier * 1.1
    r4_1d = close_1d + width_1d * camarilla_multiplier * 2.0  # R4 for exit
    s4_1d = close_1d - width_1d * camarilla_multiplier * 2.0  # S4 for exit
    
    # Align Camarilla levels to 12h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(34, 20)  # EMA34, volume MA20
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or np.isnan(r4_1d_aligned[i]) or 
            np.isnan(s4_1d_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_confirm = volume_spike[i]
        price = close[i]
        ema_trend = ema_34_1d_aligned[i]
        r3 = r3_1d_aligned[i]
        s3 = s3_1d_aligned[i]
        r4 = r4_1d_aligned[i]
        s4 = s4_1d_aligned[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above R3 with volume spike and price above 1d EMA34
            if price > r3 and vol_confirm and price > ema_trend:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short entry: price breaks below S3 with volume spike and price below 1d EMA34
            elif price < s3 and vol_confirm and price < ema_trend:
                signals[i] = -0.25
                position = -1
                entry_price = price
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit on R4 touch or volume drying up
            # Exit when price touches R4 (strong resistance) or volume dries up (< 0.5x average)
            volume_dry = volume[i] < 0.5 * volume_ma_20[i] if not np.isnan(volume_ma_20[i]) else False
            if price >= r4 or volume_dry:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit on S4 touch or volume drying up
            # Exit when price touches S4 (strong support) or volume dries up (< 0.5x average)
            volume_dry = volume[i] < 0.5 * volume_ma_20[i] if not np.isnan(volume_ma_20[i]) else False
            if price <= s4 or volume_dry:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals