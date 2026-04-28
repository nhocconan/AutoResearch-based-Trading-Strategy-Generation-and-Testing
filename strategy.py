#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation
# Uses 4h EMA50 for trend direction. Breaks above/below Camarilla R3/S3 levels on 1h
# with volume spike provide high-probability entries. Session filter (08-20 UTC) reduces noise.
# Target: 60-150 total trades over 4 years via tight breakout conditions + volume + trend filter + session.
# 4h EMA50 avoids whipsaws in ranging markets while capturing trend shifts.

name = "1h_Camarilla_R3_S3_Breakout_4hEMA50_Trend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate EMA50 on 4h close for trend filter
    close_4h = pd.Series(df_4h['close'])
    ema50_4h = close_4h.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Camarilla levels (R3, S3) on 1h data using typical price
    typical_price = (high + low + close) / 3.0
    typical_series = pd.Series(typical_price)
    typical_ma_5 = typical_series.rolling(window=5, min_periods=5).mean().values
    typical_std_5 = typical_series.rolling(window=5, min_periods=5).std().values
    camarilla_r3 = typical_ma_5 + 1.5 * typical_std_5
    camarilla_s3 = typical_ma_5 - 1.5 * typical_std_5
    
    # Align 4h EMA50 to 1h timeframe (completed 4h candles only)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(20, 50)  # Volume MA20 and 4h EMA50 need sufficient history
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(ema50_4h_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_confirm = volume_spike[i]
        session_ok = in_session[i]
        price = close[i]
        r3 = camarilla_r3[i]
        s3 = camarilla_s3[i]
        ema50_val = ema50_4h_aligned[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Camarilla R3 AND 4h EMA50 uptrend AND volume spike AND session
            if price > r3 and price > ema50_val and vol_confirm and session_ok:
                signals[i] = 0.20
                position = 1
                entry_price = price
            # Short entry: price breaks below Camarilla S3 AND 4h EMA50 downtrend AND volume spike AND session
            elif price < s3 and price < ema50_val and vol_confirm and session_ok:
                signals[i] = -0.20
                position = -1
                entry_price = price
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit on stoploss or price falls below Camarilla S3 (reversal)
            # ATR-based stoploss: 2.0 * ATR below entry (using 1h ATR)
            tr1 = high[max(0, i-1):i+1] - low[max(0, i-1):i+1]
            tr2 = np.abs(high[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr3 = np.abs(low[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            atr_val = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
            stop_loss = entry_price - 2.0 * atr_val
            # Exit on stoploss or price < s3 (reversal below support)
            if price < stop_loss or price < s3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # Short - exit on stoploss or price rises above Camarilla R3 (reversal)
            # ATR-based stoploss: 2.0 * ATR above entry
            tr1 = high[max(0, i-1):i+1] - low[max(0, i-1):i+1]
            tr2 = np.abs(high[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr3 = np.abs(low[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            atr_val = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
            stop_loss = entry_price + 2.0 * atr_val
            # Exit on stoploss or price > r3 (reversal above resistance)
            if price > stop_loss or price > r3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals