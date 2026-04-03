#!/usr/bin/env python3
"""
Experiment #1214: 1h Donchian(20) Breakout + 4h/1d Trend + Volume Confirmation
HYPOTHESIS: Donchian(20) breakouts on 1h timeframe capture short-term trends with optimal trade frequency. 
Trend filters from 4h (direction) and 1d (regime) ensure alignment with higher-timeframe momentum. 
Volume confirmation (>1.5x average) filters for institutional participation. Session filter (08-20 UTC) reduces noise. 
Designed to work in both bull (breakouts continue) and bear (breakdowns continue) markets by following the 4h trend direction. 
Target: 60-150 total trades over 4 years = 15-37/year for 1h. HARD MAX: 200 total trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1214_1h_donchian20_4h_1d_trend_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Pre-compute session hours (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    # === HTF: 4h data for trend filter (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    # Simple trend: price > previous close = uptrend, < = downtrend
    trend_4h = np.zeros(len(close_4h))
    trend_4h[1:] = np.where(close_4h[1:] > close_4h[:-1], 1, -1)
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # === HTF: 1d data for regime filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # Regime: price above/below 50-period EMA
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    regime_1d = np.zeros(len(close_1d))
    regime_1d[50:] = np.where(close_1d[50:] > ema_50_1d[50:], 1, -1)  # 1= bull regime, -1= bear regime
    regime_1d_aligned = align_htf_to_ltf(prices, df_1d, regime_1d)
    
    # === 1h Indicators: Donchian(20) ===
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 1h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 1h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size (discrete level to minimize churn)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # sufficient for 1d EMA(50) and Donchian/volume MA
    
    for i in range(warmup, n):
        # --- Session Filter: Only trade 08-20 UTC ---
        if hours[i] < 8 or hours[i] > 20:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(trend_4h_aligned[i]) or np.isnan(regime_1d_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
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
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Breakout: price breaks above upper band OR below lower band
            # Long: price > Donchian high AND 4h uptrend AND 1d bull regime
            if price > donch_high[i] and trend_4h_aligned[i] > 0 and regime_1d_aligned[i] > 0:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: price < Donchian low AND 4h downtrend AND 1d bear regime
            elif price < donch_low[i] and trend_4h_aligned[i] < 0 and regime_1d_aligned[i] < 0:
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