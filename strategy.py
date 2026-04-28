#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 12h EMA50 trend filter and volume confirmation
# Uses Camarilla pivot levels calculated from 1d data (R1/S1 for mean reversion fade, R3/S3 for breakout confirmation)
# Only takes breakout trades in direction of 12h EMA50 trend with volume confirmation (>1.8x average)
# Designed to work in both bull and bear markets by combining pivot structure with trend filter
# Target: 20-50 trades/year via tight Camarilla breakout conditions + volume + trend filter

name = "4h_Camarilla_R1S1_Breakout_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from 1d OHLC
    # Camarilla: R1 = C + ((H-L)*1.1/12), S1 = C - ((H-L)*1.1/12)
    # Using previous day's OHLC for current day's levels (no look-ahead)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift by 1 to use previous day's data (avoid look-ahead)
    high_1d_prev = np.roll(high_1d, 1)
    low_1d_prev = np.roll(low_1d, 1)
    close_1d_prev = np.roll(close_1d, 1)
    # First value will be invalid (rolled from last), but min_periods will handle alignment
    
    # Calculate Camarilla levels
    camarilla_r1 = close_1d_prev + ((high_1d_prev - low_1d_prev) * 1.1 / 12)
    camarilla_s1 = close_1d_prev - ((high_1d_prev - low_1d_prev) * 1.1 / 12)
    camarilla_r3 = close_1d_prev + ((high_1d_prev - low_1d_prev) * 1.1 / 4)
    camarilla_s3 = close_1d_prev - ((high_1d_prev - low_1d_prev) * 1.1 / 4)
    
    # Align 1d Camarilla levels to 4h timeframe (completed 1d candles only)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate EMA50 on 12h close for trend filter
    close_12h = pd.Series(df_12h['close'])
    ema50_12h = close_12h.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA50 to 4h timeframe (completed 12h candles only)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume confirmation: >1.8x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.8 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(20, 50)  # Need sufficient history for volume MA and EMA50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(ema50_12h_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_confirm = volume_spike[i]
        price = close[i]
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        ema50_val = ema50_12h_aligned[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long breakout: price breaks above R3 AND 12h EMA50 uptrend AND volume spike
            if price > r3 and price > ema50_val and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short breakout: price breaks below S3 AND 12h EMA50 downtrend AND volume spike
            elif price < s3 and price < ema50_val and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = price
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit on stoploss or price falls below R1 (mean reversion)
            # ATR-based stoploss: 2.0 * ATR below entry (using 4h ATR)
            tr1 = high[max(0, i-1):i+1] - low[max(0, i-1):i+1]
            tr2 = np.abs(high[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr3 = np.abs(low[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            atr_val = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
            stop_loss = entry_price - 2.0 * atr_val
            # Exit on stoploss or price < R1 (mean reversion back to pivot)
            if price < stop_loss or price < r1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit on stoploss or price rises above S1 (mean reversion)
            # ATR-based stoploss: 2.0 * ATR above entry
            tr1 = high[max(0, i-1):i+1] - low[max(0, i-1):i+1]
            tr2 = np.abs(high[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr3 = np.abs(low[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            atr_val = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
            stop_loss = entry_price + 2.0 * atr_val
            # Exit on stoploss or price > S1 (mean reversion back to pivot)
            if price > stop_loss or price > s1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals