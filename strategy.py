#!/usr/bin/env python3
"""
Experiment #714: 1h EMA Pullback with 4h/1d Trend Filter and Volume Spike
HYPOTHESIS: Buy pullbacks to 21 EMA in 4h/1d uptrend with volume spike (>2.0x average);
Sell rallies to 21 EMA in 4h/1d downtrend with volume spike. Uses 08-20 UTC session filter
to avoid low-liquidity hours. Target: 75-150 total trades over 4 years (19-37/year).
Position size fixed at 0.20 to minimize fee churn. Stoploss at 2*ATR.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_714_1h_ema_pullback_4h_1d_trend_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 4h and 1d data for trend filters (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 21 EMA on 4h close
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=21, min_periods=21, adjust=False).mean().values
    
    # Calculate 21 EMA on 1d close
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=21, min_periods=21, adjust=False).mean().values
    
    # Align HTF EMAs to 1h timeframe (with shift(1) for completed bars only)
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === 1h Indicators: EMA(21), ATR(14), Volume MA(20) ===
    ema_21 = pd.Series(close).ewm(span=21, min_periods=21, adjust=False).mean().values
    
    # True Range and ATR
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Volume MA(20) for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # Precompute session hours (08-20 UTC) using DatetimeIndex
    hours = prices.index.hour  # prices.index is DatetimeIndex from parquet
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = max(21, 20)  # sufficient for EMA and volume MA
    
    for i in range(warmup, n):
        # --- Session Filter: Only trade 08-20 UTC ---
        if hours[i] < 8 or hours[i] > 20:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(ema_21[i]) or np.isnan(atr[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
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
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume spike filter: require > 2.0x average volume
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike:
            # Get trend from 4h and 1d EMA alignment
            # Uptrend: price > both EMAs
            # Downtrend: price < both EMAs
            is_uptrend = price > ema_4h_aligned[i] and price > ema_1d_aligned[i]
            is_downtrend = price < ema_4h_aligned[i] and price < ema_1d_aligned[i]
            
            # Long: Pullback to EMA in uptrend with volume spike
            # Allow small tolerance (0.1%) for EMA touch
            ema_touch_long = abs(price - ema_21[i]) / ema_21[i] < 0.001
            if is_uptrend and ema_touch_long:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: Rally to EMA in downtrend with volume spike
            elif is_downtrend and ema_touch_long:  # same tolerance for EMA touch
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