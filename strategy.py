#!/usr/bin/env python3
"""
Experiment #334: 4h/1d Trend + 1h Camarilla Pivot + Volume Spike

HYPOTHESIS: Use 4h EMA50 and 1d Donchian20 for trend direction, then on 1h timeframe enter
mean reversion at Camarilla S3/R3 levels with volume confirmation (>1.5x 20-bar average).
Only trade during 08-20 UTC session to reduce noise. Target: 60-150 total trades over 4 years
(15-37/year) on 1h timeframe. Uses discrete position sizing (0.20) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_1d_trend_1h_camarilla_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # === HTF: 4h EMA50 for trend ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) >= 50:
        close_4h = df_4h['close'].values
        ema_50_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    else:
        ema_50_4h_aligned = np.full(n, np.nan)
    
    # === HTF: 1d Donchian20 for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 20:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        # Donchian channels: upper = max(high,20), lower = min(low,20)
        donch_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
        donch_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
        donch_high_20_aligned = align_htf_to_ltf(prices, df_1d, donch_high_20)
        donch_low_20_aligned = align_htf_to_ltf(prices, df_1d, donch_low_20)
    else:
        donch_high_20_aligned = np.full(n, np.nan)
        donch_low_20_aligned = np.full(n, np.nan)
    
    # === 1h Indicators ===
    # Calculate ATR(14) for stoploss
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Calculate Camarilla pivot levels for each 1h bar using prior 1h OHLC
    camarilla_s3 = np.full(n, np.nan)
    camarilla_r3 = np.full(n, np.nan)
    
    # For each 1h bar, use previous 1h bar's OHLC
    for i in range(1, n):
        ph = high[i-1]
        pl = low[i-1]
        pc = close[i-1]
        
        range_ = ph - pl
        camarilla_s3[i] = pc - range_ * 1.1 / 4
        camarilla_r3[i] = pc + range_ * 1.1 / 4
    
    # For first bar, use same values (will be filtered by warmup)
    camarilla_s3[0] = camarilla_s3[1] if n > 1 else 0
    camarilla_r3[0] = camarilla_r3[1] if n > 1 else 0
    
    # Volume ratio: current vs 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # Discrete position sizing (20% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Session Filter: Only trade 08-20 UTC ---
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(donch_high_20_aligned[i]) or 
            np.isnan(donch_low_20_aligned[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(camarilla_r3[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # --- Trend Direction from HTF ---
        # 4h EMA50: price > EMA50 = uptrend, price < EMA50 = downtrend
        # 1d Donchian: price > upper = bullish bias, price < lower = bearish bias
        price_above_4h_ema = close[i] > ema_50_4h_aligned[i]
        price_below_4h_ema = close[i] < ema_50_4h_aligned[i]
        price_above_1d_donch = close[i] > donch_high_20_aligned[i]
        price_below_1d_donch = close[i] < donch_low_20_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # ATR stoploss: 2.5 * ATR
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at opposite Camarilla level (S3 for long, R3 for short)
                if close[i] >= camarilla_r3[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at opposite Camarilla level
                if close[i] <= camarilla_s3[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Uptrend bias + price at S3 (mean reversion) with volume spike
        long_condition = (
            price_above_4h_ema and price_above_1d_donch and  # Uptrend bias from both HTF
            close[i] <= camarilla_s3[i] * 1.001 and          # Price at or below S3
            volume_spike                                     # Volume confirmation
        )
        
        # Short: Downtrend bias + price at R3 (mean reversion) with volume spike
        short_condition = (
            price_below_4h_ema and price_below_1d_donch and  # Downtrend bias from both HTF
            close[i] >= camarilla_r3[i] * 0.999 and          # Price at or above R3
            volume_spike                                     # Volume confirmation
        )
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals