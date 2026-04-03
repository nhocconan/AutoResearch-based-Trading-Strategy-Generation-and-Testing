#!/usr/bin/env python3
"""
Experiment #1587: 6h Camarilla Pivot + Volume Spike + ADX Regime
HYPOTHESIS: 6h Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) combined with volume confirmation (>1.5x average) and ADX regime filter (ADX>25 for trending, ADX<20 for ranging) captures high-probability reversals and breakouts. The 6h timeframe balances trade frequency and signal quality, while daily pivot levels provide institutional reference points. Volume spike ensures participation, and ADX filter avoids whipsaws in low-momentum environments. Target: 75-150 total trades over 4 years (19-37/year) with discrete position sizing to minimize fee churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1587_6h_camarilla_pivot_vol_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivot levels (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels from previous day
    # PP = (H + L + C) / 3
    # R4 = PP + ((H - L) * 1.1 / 2)
    # R3 = PP + ((H - L) * 1.1 / 4)
    # S3 = PP - ((H - L) * 1.1 / 4)
    # S4 = PP - ((H - L) * 1.1 / 2)
    pp = (high_1d + low_1d + close_1d) / 3.0
    r4 = pp + ((high_1d - low_1d) * 1.1 / 2.0)
    r3 = pp + ((high_1d - low_1d) * 1.1 / 4.0)
    s3 = pp - ((high_1d - low_1d) * 1.1 / 4.0)
    s4 = pp - ((high_1d - low_1d) * 1.1 / 2.0)
    
    # Align HTF pivot levels to 6h timeframe (shifted by 1 for completed bars only)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ADX(14) for regime filter ===
    # True Range
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    
    # Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        plus_dm[i] = up_move if up_move > down_move and up_move > 0 else 0
        minus_dm[i] = down_move if down_move > up_move and down_move > 0 else 0
    
    # Smoothed values
    tr_ma = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    plus_dm_ma = pd.Series(plus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values
    minus_dm_ma = pd.Series(minus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_ma / tr_ma
    minus_di = 100 * minus_dm_ma / tr_ma
    
    # DX and ADX
    dx = np.zeros(n)
    dx[14:] = 100 * np.abs(plus_di[14:] - minus_di[14:]) / (plus_di[14:] + minus_di[14:])
    adx = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 20  # sufficient for volume MA and ADX
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(r4_6h[i]) or np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or np.isnan(s4_6h[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(adx[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: ATR-based stoploss (using TR as proxy for simplicity) ---
        if in_position:
            bars_since_entry += 1
            
            # Calculate current ATR (14) for stoploss
            if i >= 14:
                atr_val = tr_ma[i]
            else:
                atr_val = np.mean(tr[max(0, i-13):i+1])
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr_val
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.0 * atr_val
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        # ADX regime filter: ADX > 25 for trending (breakout), ADX < 20 for ranging (mean reversion)
        strong_trend = adx[i] > 25
        ranging = adx[i] < 20
        
        if volume_spike:
            # In strong trend: look for breakouts at R4/S4
            if strong_trend:
                if price > r4_6h[i]:  # Bullish breakout
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
                elif price < s4_6h[i]:  # Bearish breakdown
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = -SIZE
                else:
                    signals[i] = 0.0
            # In ranging market: look for mean reversion at R3/S3
            elif ranging:
                if price < r3_6h[i] and price > s3_6h[i]:  # Within range, look for reversals
                    if price <= s3_6h[i] * 1.002:  # Near support, go long
                        in_position = True
                        position_side = 1
                        entry_price = close[i]
                        bars_since_entry = 0
                        signals[i] = SIZE
                    elif price >= r3_6h[i] * 0.998:  # Near resistance, go short
                        in_position = True
                        position_side = -1
                        entry_price = close[i]
                        bars_since_entry = 0
                        signals[i] = -SIZE
                    else:
                        signals[i] = 0.0
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals