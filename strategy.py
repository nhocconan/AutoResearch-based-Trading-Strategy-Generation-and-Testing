#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume spike
# Long when price breaks above R3 AND close > 4h EMA50 AND volume > 2x 20-bar avg
# Short when price breaks below S3 AND close < 4h EMA50 AND volume > 2x 20-bar avg
# Exit when price returns to Camarilla H/L levels or volume drops
# Target: 15-37 trades/year via tight Camarilla breakout conditions + volume confirmation
# Uses 4h for trend direction (EMA50), 1h for precise entry timing
# Works in bull markets via breakouts, bear markets via short breakdowns

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_Trend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    # Calculate EMA(50) on 4h close
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h EMA50 to 1h timeframe
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla levels on 1h data (using previous bar's OHLC)
    # Camarilla: H4 = Close + 1.1*(High-Low)*1.1/2, L4 = Close - 1.1*(High-Low)*1.1/2
    # R3 = Close + 1.1*(High-Low)*1.1/4, S3 = Close - 1.1*(High-Low)*1.1/4
    # Actually standard Camarilla: R4 = Close + 1.1*(High-Low)*1.1/2, R3 = Close + 1.1*(High-Low)*1.1/4
    # We'll use typical Camarilla calculation based on previous bar
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]  # Fill first value
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    camarilla_range = prev_high - prev_low
    r3 = prev_close + 1.1 * camarilla_range * 1.1 / 4
    s3 = prev_close - 1.1 * camarilla_range * 1.1 / 4
    h3 = prev_close + 1.1 * camarilla_range * 1.1 / 6  # For exit
    l3 = prev_close - 1.1 * camarilla_range * 1.1 / 6  # For exit
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Need sufficient history for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(volume_ma_20[i]) or 
            np.isnan(r3[i]) or np.isnan(s3[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        ema_50 = ema_50_4h_aligned[i]
        price = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when price breaks above R3 AND close > 4h EMA50 AND volume confirmation
            if high[i] > r3[i] and price > ema_50 and vol_conf:
                signals[i] = 0.20
                position = 1
            # Short when price breaks below S3 AND close < 4h EMA50 AND volume confirmation
            elif low[i] < s3[i] and price < ema_50 and vol_conf:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when price returns to H3 or below or volume drops
            if low[i] < h3[i] or not vol_conf:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # Short - exit when price returns to L3 or above or volume drops
            if high[i] > l3[i] or not vol_conf:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals