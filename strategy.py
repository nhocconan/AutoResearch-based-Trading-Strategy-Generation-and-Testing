#!/usr/bin/env python3
"""
Experiment #5754: 1h RSI(14) pullback + 4h EMA(50) trend + 1d volume spike + session filter
HYPOTHESIS: In 1h timeframe, buy pullbacks to EMA50 on 4h uptrend with volume confirmation on 1d, sell rallies to EMA50 on 4h downtrend. Uses RSI(14) < 40 for long entry and > 60 for short entry to catch mean reversion within trend. Session filter (08-20 UTC) avoids low liquidity. Discrete sizing 0.20 minimizes fee churn. Designed to work in both bull (trend-following pulls back) and bear (counter-trend bounces fade) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5754_1h_rsi_pullback_4h_ema50_1d_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # === HTF: 4h data for EMA50 trend ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) >= 50:
        close_4h = df_4h['close'].values
        ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    else:
        ema_4h = np.full(len(df_4h), np.nan)
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # === HTF: 1d data for volume spike ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 20:
        volume_1d = df_1d['volume'].values
        avg_volume_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
        volume_ratio_1d = volume_1d / np.where(avg_volume_1d > 0, avg_volume_1d, 1)
    else:
        volume_ratio_1d = np.full(len(df_1d), np.nan)
    volume_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ratio_1d)
    
    # === 1h Indicators: RSI(14) ===
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    # === 1h Indicators: ATR(14) for stoploss ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size (discrete level)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(14, 50, 20, 14)  # RSI, EMA50, volume avg, ATR
    
    for i in range(warmup, n):
        # --- Session Filter: Trade only during active UTC hours ---
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(rsi[i]) or np.isnan(ema_4h_aligned[i]) or 
            np.isnan(volume_ratio_1d_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                stop_price = entry_price - 2.5 * atr[i]
                if price <= stop_price:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                stop_price = entry_price + 2.5 * atr[i]
                if price >= stop_price:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Trend filter: 4h EMA50
        uptrend = price > ema_4h_aligned[i]
        downtrend = price < ema_4h_aligned[i]
        
        # Volume confirmation: 1d volume spike (> 1.5x average)
        volume_spike = volume_ratio_1d_aligned[i] > 1.5
        
        # RSI conditions for pullback entry
        rsi_oversold = rsi[i] < 40
        rsi_overbought = rsi[i] > 60
        
        # Entry: long on pullback in uptrend, short on rally in downtrend
        long_setup = uptrend and volume_spike and rsi_oversold
        short_setup = downtrend and volume_spike and rsi_overbought
        
        if long_setup:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_setup:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals