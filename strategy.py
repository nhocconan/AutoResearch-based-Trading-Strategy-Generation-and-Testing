#!/usr/bin/env python3
"""
1d_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike
Hypothesis: On daily timeframe, Camarilla R3/S3 level breakouts filtered by weekly trend (price > weekly EMA34) and volume spikes capture institutional moves with controlled frequency. Long when price breaks above R3 in bullish weekly trend with volume confirmation; short when price breaks below S3 in bearish weekly trend with volume confirmation. Uses discrete sizing (±0.30) and close-based exits to target 15-25 trades/year. Works in both bull/bear markets by only trading in direction of higher-timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Weekly EMA34 for higher-timeframe trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Daily Camarilla levels (based on previous day's range)
    # Calculate pivot and levels using prior day's OHLC
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    
    # First bar will have invalid prev values, but we have warmup
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    r3 = pivot + (range_hl * 1.1 / 4)  # R3 = Close + [(High-Low) * 1.1/4]
    s3 = pivot - (range_hl * 1.1 / 4)  # S3 = Close - [(High-Low) * 1.1/4]
    r4 = pivot + (range_hl * 1.1 / 2)  # R4 = Close + [(High-Low) * 1.1/2]
    s4 = pivot - (range_hl * 1.1 / 2)  # S4 = Close - [(High-Low) * 1.1/2]
    
    # Volume spike detection (20-period volume SMA)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_sma_20 * 2.0)  # Volume > 2x 20-period average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.30
    
    # Warmup: need prior day data (1) + weekly EMA34 (34) + volume SMA20 (20)
    start_idx = max(1, 34, 20) + 1  # +1 to ensure weekly bar completion
    
    for i in range(start_idx, n):
        # Skip if any data not ready (NaN from calculation)
        if (np.isnan(r3[i]) or np.isnan(s3[i]) or
            np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(vol_sma_20[i])):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        close_val = close[i]
        r3_val = r3[i]
        s3_val = s3[i]
        ema_34_val = ema_34_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine weekly trend: bullish if price > EMA34, bearish if price < EMA34
        bullish_1w = close_val > ema_34_val
        bearish_1w = close_val < ema_34_val
        
        # Entry conditions: price breaks above/below Camarilla R3/S3 in direction of weekly trend with volume confirmation
        long_entry = (close_val > r3_val) and bullish_1w and vol_spike
        short_entry = (close_val < s3_val) and bearish_1w and vol_spike
        
        # Exit conditions: price returns to pivot level or trend reversal
        long_exit = (close_val < pivot[i]) or not bullish_1w
        short_exit = (close_val > pivot[i]) or not bearish_1w
        
        # Simplified exit: flip signal on opposite condition or pivot re-entry
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and (close_val < pivot[i] or not bullish_1w):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (close_val > pivot[i] or not bearish_1w):
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "1d_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0