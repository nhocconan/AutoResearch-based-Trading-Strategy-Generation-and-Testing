#!/usr/bin/env python3
"""
Experiment #294: 1h Camarilla Pivot Long + 4h/1d Regime Filter + Volume Spike

HYPOTHESIS: 1h long entries at Camarilla L3 support with 4h/1d trend alignment (price > EMA50) 
and volume spikes (>1.8x average) capture high-probability bounces in both bull and bear markets. 
The 4h EMA50 provides intermediate trend filter, 1d EMA200 provides long-term bias, reducing 
false signals during strong counter-trend moves. Shorts are avoided due to 1h timeframe's 
vulnerability to bearish whipsaws; only longs are taken with strict filters. 
Session filter (08-20 UTC) reduces noise outside active trading hours. 
Target: 60-150 total trades over 4 years = 15-37/year for 1h. Uses discrete position sizing (0.20) 
to minimize churn. ATR-based stoploss (2.5x) manages risk.

IMPLEMENTATION NOTES:
- Uses discrete position sizing (0.20) to minimize churn
- Volume confirmation threshold set to 1.8x to balance signal quality and frequency
- Minimum holding period of 2 bars to reduce churn
- Warmup period set to 100 bars for stable indicators
- Only long positions taken (shorts disabled for 1h robustness in bear markets)
- Exits on Camarilla H3 reversion (mean reversion target) OR ATR stoploss
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_294_1h_camarilla_4h_1d_regime_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === Precompute session hours (08-20 UTC) ===
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # === HTF: 4h data for EMA50 trend (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    ema_4h = pd.Series(df_4h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # === HTF: 1d data for EMA200 trend (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === 1h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 1h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === 1h Indicators: Camarilla Pivot Levels (based on previous bar) ===
    camarilla_h3 = np.full(n, np.nan)
    camarilla_l3 = np.full(n, np.nan)
    camarilla_h4 = np.full(n, np.nan)
    camarilla_l4 = np.full(n, np.nan)
    
    for i in range(1, n):
        # Calculate pivot from previous bar (i-1)
        high_prev = high[i-1]
        low_prev = low[i-1]
        close_prev = close[i-1]
        
        pivot = (high_prev + low_prev + close_prev) / 3.0
        range_prev = high_prev - low_prev
        
        # Camarilla levels
        camarilla_h3[i] = pivot + (range_prev * 1.1 / 4)
        camarilla_l3[i] = pivot - (range_prev * 1.1 / 4)
        camarilla_h4[i] = pivot + (range_prev * 1.1 / 2)
        camarilla_l4[i] = pivot - (range_prev * 1.1 / 2)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # Position sizing (20% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0  # Track bars in position for minimum holding period
    
    warmup = 100  # Increased warmup for stable HTF alignment and indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i]) or 
            np.isnan(camarilla_l3[i]) or np.isnan(camarilla_h3[i])):
            signals[i] = 0.0
            continue
        
        # --- Session Filter: 08-20 UTC only ---
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: 4h EMA50 > 1d EMA200 = bullish alignment ---
        bullish_regime = ema_4h_aligned[i] > ema_1d_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.8x average) ---
        volume_spike = vol_ratio[i] > 1.8
        
        # --- Camarilla L3 Long Entry: Price at L3 support with confirmation ---
        long_entry = (low[i] <= camarilla_l3[i]) and bullish_regime and volume_spike
        
        # --- Exit Logic ---
        if in_position:
            bars_since_entry += 1
            
            # ATR-based stoploss (2.5x ATR)
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                
                # Exit on Camarilla H3 reversion (mean reversion target)
                if high[i] >= camarilla_h3[i]:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 2 bars to reduce churn
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long only: Camarilla L3 bounce with regime and volume confirmation
        if long_entry:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        else:
            signals[i] = 0.0
    
    return signals