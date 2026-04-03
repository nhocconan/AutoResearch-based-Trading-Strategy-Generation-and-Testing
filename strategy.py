#!/usr/bin/env python3
"""
Experiment #154: 1h RSI(14) pullback + 4h EMA(50) trend + 1d volume confirmation + ATR stoploss
HYPOTHESIS: In 1h timeframe, RSI pullbacks during strong 4h/1d trends with volume confirmation capture swing trades with controlled frequency. The 4h EMA(50) provides medium-term trend bias, 1d volume ensures institutional participation, and RSI(14)<30/>70 identifies exhaustion points. ATR-based stops limit losses. Target: 60-150 total trades over 4 years (15-37/year) for 1h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_154_1h_rsi14_pullback_4h_ema50_1d_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Pre-compute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(open_time).hour
    
    # === HTF: 4h data for EMA(50) trend (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = pd.Series(df_4h['close'].values)
    ema_50_4h = close_4h.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_trend_4h = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # === HTF: 1d data for volume MA(20) confirmation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    volume_1d = pd.Series(df_1d['volume'].values)
    vol_ma_20_1d = volume_1d.rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned_1d = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # === 1h Indicators: RSI(14) ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0.0)
    rsi = 100 - (100 / (1 + rs))
    
    # === 1h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 100  # sufficient for RSI/ATR warmup + HTF
    
    for i in range(warmup, n):
        # --- Session Filter: 08-20 UTC only ---
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(rsi[i]) or np.isnan(ema_trend_4h[i]) or
            np.isnan(vol_ma_aligned_1d[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require current volume > 1.5x 1d average ---
        volume_spike = volume[i] > 1.5 * vol_ma_aligned_1d[i]
        
        # --- RSI Conditions ---
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        # --- 4h EMA Trend ---
        bullish_trend = close[i] > ema_trend_4h[i]
        bearish_trend = close[i] < ema_trend_4h[i]
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.0*ATR below entry
                stop_level = entry_price - 2.0 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.0*ATR above entry
                stop_level = entry_price + 2.0 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Optional: time-based exit after 24 bars (~24h on 1h) to avoid overtrading
            if bars_since_entry > 24:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        if volume_spike:
            # Long: RSI oversold AND bullish 4h trend
            if rsi_oversold and bullish_trend:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: RSI overbought AND bearish 4h trend
            elif rsi_overbought and bearish_trend:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals