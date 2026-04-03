#!/usr/bin/env python3
"""
Experiment #034: 1h RSI(14) mean reversion + 4h/1d trend filter + volume confirmation + session filter
HYPOTHESIS: In ranging markets (common in 2025 BTC/ETH), price reverts to the mean after extreme RSI readings. 
We use 4h EMA(50) and 1d EMA(200) for trend alignment (only trade pullbacks in trend direction) and 
volume confirmation to avoid false signals. Session filter (08-20 UTC) reduces noise. 
Target: 60-150 total trades over 4 years (15-37/year) with discrete sizing (0.20).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_034_1h_rsi14_meanrev_4h_ema50_1d_ema200_vol_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 4h EMA(50) for intermediate trend (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # === HTF: 1d EMA(200) for long-term trend (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    ema_200_1d = pd.Series(df_1d['close'].values).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # === 1h Indicators: RSI(14) for mean reversion ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain_ma = pd.Series(gain).ewm(span=14, min_periods=14, adjust=False).mean().values
    loss_ma = pd.Series(loss).ewm(span=14, min_periods=14, adjust=False).mean().values
    rs = np.divide(gain_ma, loss_ma, out=np.zeros_like(gain_ma), where=loss_ma!=0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    # === 1h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    valid_start = 20
    vol_ratio[valid_start:] = volume[valid_start:] / vol_ma[valid_start:]
    vol_ratio[:valid_start] = 1.0
    
    # === Session filter: 08-20 UTC (pre-compute hours array) ===
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 200  # sufficient for 1d EMA200 calculation
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(rsi[i]) or np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(ema_200_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Session Filter: Only trade 08-20 UTC ---
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Trend Alignment: Only trade pullbacks in trend direction ---
        # Bullish trend: price above both 4h EMA50 and 1d EMA200
        bullish_trend = price > ema_50_4h_aligned[i] and price > ema_200_1d_aligned[i]
        # Bearish trend: price below both 4h EMA50 and 1d EMA200
        bearish_trend = price < ema_50_4h_aligned[i] and price < ema_200_1d_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.3x average) ---
        volume_spike = vol_ratio[i] > 1.3
        
        # --- Mean Reversion Entry: Extreme RSI readings ---
        # Long: RSI < 30 (oversold) in bullish trend
        # Short: RSI > 70 (overbought) in bearish trend
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        # --- Exit Logic: Mean reversion to midpoint (RSI 50) ---
        if in_position:
            if position_side > 0:  # Long position
                # Exit when RSI returns to 50 (mean reversion complete)
                if rsi[i] >= 50:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                # Exit when RSI returns to 50 (mean reversion complete)
                if rsi[i] <= 50:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        if volume_spike:
            # Long: oversold RSI AND bullish trend alignment
            if rsi_oversold and bullish_trend:
                in_position = True
                position_side = 1
                signals[i] = SIZE
            # Short: overbought RSI AND bearish trend alignment
            elif rsi_overbought and bearish_trend:
                in_position = True
                position_side = -1
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals