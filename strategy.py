#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Uses discrete sizing 0.25 to minimize fee churn. Target: 50-150 trades over 4 years (12-37/year).
# Works in bull markets (breakouts with trend) and bear markets (fade at extremes in range).
# Camarilla levels from 1d provide institutional support/resistance; volume confirms conviction.
# Focus on BTC/ETH as primary symbols.

name = "6h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Camarilla pivot levels (R3, R4, S3, S4)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    # Use previous day's OHLC for today's Camarilla levels
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    r3 = pivot + (range_hl * 1.1 / 2.0)  # R3 = pivot + 1.1*(H-L)/2
    r4 = pivot + (range_hl * 1.1)        # R4 = pivot + 1.1*(H-L)
    s3 = pivot - (range_hl * 1.1 / 2.0)  # S3 = pivot - 1.1*(H-L)/2
    s4 = pivot - (range_hl * 1.1)        # S4 = pivot - 1.1*(H-L)
    
    # Align Camarilla levels to 6h timeframe (wait for 1d bar close)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 2.0x 20-period average (strict to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(100, 34, 20)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_r3 = r3_aligned[i]
        curr_r4 = r4_aligned[i]
        curr_s3 = s3_aligned[i]
        curr_s4 = s4_aligned[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        curr_volume_spike = volume_spike[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade on volume spike
            if curr_volume_spike:
                # Bullish breakout: Close breaks above R3 with trend filter (price > EMA34)
                if curr_close > curr_r3 and curr_close > curr_ema_34_1d:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish breakout: Close breaks below S3 with trend filter (price < EMA34)
                elif curr_close < curr_s3 and curr_close < curr_ema_34_1d:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
                # Bullish fade: Price rejects S4 (long opportunity in bearish extreme)
                elif curr_low <= curr_s4 and curr_close > curr_s4 and curr_close < curr_ema_34_1d:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish fade: Price rejects R4 (short opportunity in bullish extreme)
                elif curr_high >= curr_r4 and curr_close < curr_r4 and curr_close > curr_ema_34_1d:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit: Close drops below S3 OR loses 1d trend (close < EMA34)
            if curr_close < curr_s3 or curr_close < curr_ema_34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Close rises above R3 OR loses 1d trend (close > EMA34)
            if curr_close > curr_r3 or curr_close > curr_ema_34_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals