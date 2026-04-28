#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 Breakout with 1d Trend Filter and Volume Confirmation
# Camarilla pivots from daily data provide institutional support/resistance levels.
# Long when price breaks above R3 with volume confirmation and 1d EMA34 uptrend.
# Short when price breaks below S3 with volume confirmation and 1d EMA34 downtrend.
# Uses discrete position sizing (0.25) to minimize fee churn. Targets 12-37 trades/year.

name = "6h_Camarilla_R3_S3_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Camarilla: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), 
    #            S3 = C - ((H-L)*1.1/4), S4 = C - ((H-L)*1.1/2)
    # We need previous day's data, so shift by 1
    if len(df_1d) >= 2:
        prev_high = df_1d['high'].iloc[-2] if len(df_1d) >= 2 else df_1d['high'].iloc[-1]
        prev_low = df_1d['low'].iloc[-2] if len(df_1d) >= 2 else df_1d['low'].iloc[-1]
        prev_close = df_1d['close'].iloc[-2] if len(df_1d) >= 2 else df_1d['close'].iloc[-1]
    else:
        prev_high = df_1d['high'].iloc[-1]
        prev_low = df_1d['low'].iloc[-1]
        prev_close = df_1d['close'].iloc[-1]
    
    # Calculate Camarilla levels
    rang = prev_high - prev_low
    r3 = prev_close + (rang * 1.1 / 4)
    s3 = prev_close - (rang * 1.1 / 4)
    
    # Create arrays of same length as prices filled with Camarilla levels
    r3_array = np.full(n, r3)
    s3_array = np.full(n, s3)
    
    # Align Camarilla levels to 6h timeframe (no delay needed as they're based on completed daily bar)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3_array)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3_array)
    
    # Volume confirmation: >1.8x 20-bar average volume (tighter filter)
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.8 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(20, 34)  # volume MA(20), 1d EMA(34)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_confirm = volume_spike[i]
        price = close[i]
        curr_r3 = r3_6h[i]
        curr_s3 = s3_6h[i]
        ema_trend = ema_34_1d_aligned[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above R3 with volume spike and uptrend
            if price > curr_r3 and vol_confirm and price > ema_trend:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short entry: Price breaks below S3 with volume spike and downtrend
            elif price < curr_s3 and vol_confirm and price < ema_trend:
                signals[i] = -0.25
                position = -1
                entry_price = price
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit on stoploss or price breaks below S3
            # ATR-based stoploss: 2.5 * ATR below entry (using 6h ATR)
            tr1 = high[max(0, i-1):i+1] - low[max(0, i-1):i+1]
            tr2 = np.abs(high[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr3 = np.abs(low[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            atr_val = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
            stop_loss = entry_price - 2.5 * atr_val
            if price < stop_loss or price < curr_s3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit on stoploss or price breaks above R3
            # ATR-based stoploss: 2.5 * ATR above entry
            tr1 = high[max(0, i-1):i+1] - low[max(0, i-1):i+1]
            tr2 = np.abs(high[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr3 = np.abs(low[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            atr_val = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
            stop_loss = entry_price + 2.5 * atr_val
            if price > stop_loss or price > curr_r3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals