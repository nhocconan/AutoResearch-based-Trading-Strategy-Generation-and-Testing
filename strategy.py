#!/usr/bin/env python3
"""
Experiment #014: 1h RSI Extreme + 4h Trend Filter + 1d Volume Spike

HYPOTHESIS: RSI extremes (oversold <30 for long, overbought >70 for short) on 1h timeframe,
combined with 4h EMA50 trend filter and 1d volume spike confirmation (>2x average),
creates a strategy that captures mean reversion in trending markets with institutional participation.
Targets 15-37 trades/year on 1h (60-150 total over 4 years) by using strict confluence:
RSI extreme + HTF trend alignment + volume spike. Works in bull/bear via trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp014_1h_rsi_extreme_4h_trend_1d_vol"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 4h data for trend filter (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) >= 50:
        close_4h = df_4h['close'].values
        ema_50_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    else:
        ema_50_4h_aligned = np.full(n, np.nan)
    
    # === HTF: 1d data for volume spike (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 20:
        vol_1d = df_1d['volume'].values
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1d = np.zeros(len(vol_1d))
        vol_ratio_1d[20:] = vol_1d[20:] / vol_ma_20[20:]
        vol_ratio_1d[:20] = 1.0
        vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    else:
        vol_ratio_1d_aligned = np.full(n, 1.0)
    
    # === 1h Indicators ===
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain_ema = pd.Series(gain).ewm(span=14, min_periods=14, adjust=False).mean().values
    loss_ema = pd.Series(loss).ewm(span=14, min_periods=14, adjust=False).mean().values
    rs = gain_ema / (loss_ema + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # Discrete position sizing (20% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i]) or np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        # --- Session Filter: Only trade 08-20 UTC ---
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        if not in_session:
            signals[i] = 0.0
            continue
        
        # --- HTF Trend Filter: 4h EMA50 ---
        price_above_4h_ema = close[i] > ema_50_4h_aligned[i]
        price_below_4h_ema = close[i] < ema_50_4h_aligned[i]
        
        # --- Volume Confirmation: 1d volume spike > 2.0x average ---
        volume_spike = vol_ratio_1d_aligned[i] > 2.0
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: RSI < 30 (oversold) + price above 4h EMA50 + volume spike
        long_condition = (
            rsi[i] < 30 and price_above_4h_ema and volume_spike
        )
        
        # Short: RSI > 70 (overbought) + price below 4h EMA50 + volume spike
        short_condition = (
            rsi[i] > 70 and price_below_4h_ema and volume_spike
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