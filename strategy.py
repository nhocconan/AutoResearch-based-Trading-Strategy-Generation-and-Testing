#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h EMA20 trend filter and volume spike confirmation
# Uses Camarilla pivot levels from 1d data (R1/S1 for breakout, R4/S4 for stop levels)
# Only takes breakout trades in direction of 4h EMA20 trend with volume confirmation (>1.5x average)
# Session filter: 08-20 UTC to avoid low-liquidity hours
# Target: 15-37 trades/year via tight Camarilla breakout conditions + volume + trend filter + session
# Works in bull/bear by combining pivot structure with trend filter and volume confirmation

name = "1h_Camarilla_R1S1_Breakout_4hEMA20_Trend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute hour for session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from 1d OHLC (using previous day's data)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift by 1 to use previous day's data (avoid look-ahead)
    high_1d_prev = np.roll(high_1d, 1)
    low_1d_prev = np.roll(low_1d, 1)
    close_1d_prev = np.roll(close_1d, 1)
    # First value will be invalid (rolled from last), but min_periods will handle alignment
    
    # Calculate Camarilla levels: R1 = C + ((H-L)*1.1/12), S1 = C - ((H-L)*1.1/12)
    camarilla_r1 = close_1d_prev + ((high_1d_prev - low_1d_prev) * 1.1 / 12)
    camarilla_s1 = close_1d_prev - ((high_1d_prev - low_1d_prev) * 1.1 / 12)
    camarilla_r4 = close_1d_prev + ((high_1d_prev - low_1d_prev) * 1.1 / 2)  # Stop level
    camarilla_s4 = close_1d_prev - ((high_1d_prev - low_1d_prev) * 1.1 / 2)  # Stop level
    
    # Align 1d Camarilla levels to 1h timeframe (completed 1d candles only)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Get 4h data for EMA20 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate EMA20 on 4h close for trend filter
    close_4h = pd.Series(df_4h['close'])
    ema20_4h = close_4h.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 4h EMA20 to 1h timeframe (completed 4h candles only)
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # Volume confirmation: >1.5x 24-bar average volume (6h average)
    volume_series = pd.Series(volume)
    volume_ma_24 = volume_series.rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > 1.5 * volume_ma_24
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(24, 20)  # Need sufficient history for volume MA and EMA20
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC only
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(ema20_4h_aligned[i]) or np.isnan(volume_ma_24[i])):
            signals[i] = 0.0
            continue
        
        vol_confirm = volume_spike[i]
        price = close[i]
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        r4 = camarilla_r4_aligned[i]  # Stop level for longs
        s4 = camarilla_s4_aligned[i]  # Stop level for shorts
        ema20_val = ema20_4h_aligned[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long breakout: price breaks above R1 AND 4h EMA20 uptrend AND volume spike
            if price > r1 and price > ema20_val and vol_confirm:
                signals[i] = 0.20
                position = 1
                entry_price = price
            # Short breakout: price breaks below S1 AND 4h EMA20 downtrend AND volume spike
            elif price < s1 and price < ema20_val and vol_confirm:
                signals[i] = -0.20
                position = -1
                entry_price = price
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit on stoploss or price falls below S1 (mean reversion)
            # Stoploss at S4 level (camarilla S4)
            stop_loss = s4
            # Exit on stoploss or price < S1 (mean reversion back to pivot)
            if price < stop_loss or price < s1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # Short - exit on stoploss or price rises above S1 (mean reversion)
            # Stoploss at R4 level (camarilla R4)
            stop_loss = r4
            # Exit on stoploss or price > S1 (mean reversion back to pivot)
            if price > stop_loss or price > s1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals