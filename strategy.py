#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with weekly trend filter and volume confirmation
# - Long when price breaks above Camarilla R4 AND weekly close > weekly open (bullish weekly candle)
# - Short when price breaks below Camarilla S4 AND weekly close < weekly open (bearish weekly candle)
# - Volume confirmation: 6h volume > 2.0x 20-period 6h volume SMA
# - Exit: price retests Camarilla PP (pivot point) or opposite R4/S4 breakout with volume
# - Position sizing: 0.25 discrete level
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Weekly trend filter ensures alignment with higher timeframe momentum, reducing counter-trend whipsaw

name = "6h_1w_camarilla_breakout_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Calculate Camarilla pivot levels from daily OHLC
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return signals
    
    # Camarilla calculations: based on previous day's OHLC
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    pp = (prev_high + prev_low + prev_close) / 3.0
    r4 = pp + ((prev_high - prev_low) * 1.1 / 2)
    s4 = pp - ((prev_high - prev_low) * 1.1 / 2)
    r3 = pp + ((prev_high - prev_low) * 1.1 / 4)
    s3 = pp - ((prev_high - prev_low) * 1.1 / 4)
    
    # Align daily Camarilla levels to 6h timeframe (completed 1d bar only)
    pp_6h = align_htf_to_ltf(prices, df_1d, pp)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    
    # Weekly trend filter: bullish if weekly close > weekly open
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        weekly_bullish = np.ones(len(prices))  # default to bullish if no weekly data
        weekly_bearish = np.zeros(len(prices))
    else:
        weekly_open = df_1w['open'].values
        weekly_close = df_1w['close'].values
        weekly_bullish_raw = weekly_close > weekly_open
        weekly_bearish_raw = weekly_close < weekly_open
        weekly_bullish = align_htf_to_ltf(prices, df_1w, weekly_bullish_raw.astype(float))
        weekly_bearish = align_htf_to_ltf(prices, df_1w, weekly_bearish_raw.astype(float))
    
    # Volume confirmation: 6h volume > 2.0x 20-period volume SMA
    volume_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Track entry price for potential re-entry prevention
    entry_price = np.full(n, np.nan)
    
    for i in range(20, n):  # Start after volume SMA warmup
        # Skip if any required data is invalid
        if (np.isnan(pp_6h[i]) or np.isnan(r4_6h[i]) or np.isnan(s4_6h[i]) or
            np.isnan(volume_sma_20[i]) or np.isnan(weekly_bullish[i]) or np.isnan(weekly_bearish[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: 6h volume > 2.0x 20-period volume SMA
        vol_confirm = volume[i] > 2.0 * volume_sma_20[i]
        
        if position == 0:  # Flat - look for entry
            # Long: break above R4 with weekly bullish bias and volume
            if close[i] > r4_6h[i] and weekly_bullish[i] > 0.5 and vol_confirm:
                position = 1
                signals[i] = 0.25
                entry_price[i] = close[i]
            # Short: break below S4 with weekly bearish bias and volume
            elif close[i] < s4_6h[i] and weekly_bearish[i] > 0.5 and vol_confirm:
                position = -1
                signals[i] = -0.25
                entry_price[i] = close[i]
            else:
                signals[i] = 0.0
        elif position == 1:  # Long position - look for exit
            # Exit on retest of pivot point or opposite S4 break with volume
            exit_condition = (close[i] < pp_6h[i]) or \
                           (close[i] < s4_6h[i] and vol_confirm)
            if exit_condition:
                position = 0
                signals[i] = 0.0
                entry_price[i] = np.nan
            else:
                signals[i] = 0.25
        else:  # position == -1 (Short position) - look for exit
            # Exit on retest of pivot point or opposite R4 break with volume
            exit_condition = (close[i] > pp_6h[i]) or \
                           (close[i] > r4_6h[i] and vol_confirm)
            if exit_condition:
                position = 0
                signals[i] = 0.0
                entry_price[i] = np.nan
            else:
                signals[i] = -0.25
    
    return signals