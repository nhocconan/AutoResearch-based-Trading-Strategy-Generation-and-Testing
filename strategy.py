#!/usr/bin/env python3
"""
Experiment #174: 1h Camarilla Pivot + 4h/1d Trend + Volume Spike

HYPOTHESIS: Camarilla pivot levels on 1h provide precise entry/exit points for mean reversion in ranging markets and breakout confirmation in trending markets. Filtered by 4h HMA trend and 1d volume spike to avoid false signals. Uses discrete position sizing (0.20) and session filter (08-20 UTC) to minimize fee drag. Target: 15-37 trades/year (60-150 total over 4 years) to overcome 1h timeframe difficulties. Works in bull markets (breakouts with volume) and bear markets (mean reversion at extremes).

IMPLEMENTATION NOTES:
- Uses discrete position sizing (0.20) to minimize churn
- Camarilla levels calculated from previous 4h bar (H1/L1/C1)
- 4h HMA(21) for trend filter, 1d volume MA(20) for spike confirmation
- Minimum holding period of 2 bars to reduce churn
- Warmup period set to 100 bars for stable indicators
- Session filter: only trade 08-20 UTC
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_174_1h_camarilla_4h_1d_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(open_time).hour
    
    # === HTF: 4h data for HMA trend (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HMA(21) on 4h data
    def calculate_hma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        wma_half = pd.Series(arr).rolling(window=half_period, min_periods=half_period).mean().values
        wma_full = pd.Series(arr).rolling(window=period, min_periods=period).mean().values
        raw_hma = 2.0 * wma_half - wma_full
        hma = pd.Series(raw_hma).rolling(window=sqrt_period, min_periods=sqrt_period).mean().values
        return hma
    
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # === HTF: 1d data for volume spike (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # === 1h Indicators: Camarilla Pivot Levels ===
    # Based on previous bar's H/L/C
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    # Camarilla levels: H5, H4, H3, L3, L4, L5
    # H4 = Close + 1.1*(High-Low)/2, L4 = Close - 1.1*(High-Low)/2
    # H3 = Close + 1.1*(High-Low)/4, L3 = Close - 1.1*(High-Low)/4
    camarilla_h4 = np.full(n, np.nan)
    camarilla_l4 = np.full(n, np.nan)
    camarilla_h3 = np.full(n, np.nan)
    camarilla_l3 = np.full(n, np.nan)
    
    for i in range(1, n):
        range_val = prev_high[i] - prev_low[i]
        camarilla_h4[i] = prev_close[i] + 1.1 * range_val / 2
        camarilla_l4[i] = prev_close[i] - 1.1 * range_val / 2
        camarilla_h3[i] = prev_close[i] + 1.1 * range_val / 4
        camarilla_l3[i] = prev_close[i] - 1.1 * range_val / 4
    
    # === 1h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # Position sizing (20% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0  # Track bars in position for minimum holding period
    
    warmup = 100  # Warmup for stable HTF alignment and indicators
    
    for i in range(warmup, n):
        # --- Session Filter: Only trade 08-20 UTC ---
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(camarilla_h4[i]) or np.isnan(camarilla_l4[i]) or 
            np.isnan(hma_4h_aligned[i]) or np.isnan(atr_14[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- 4h HMA Trend Filter ---
        price_above_hma = close[i] > hma_4h_aligned[i]
        price_below_hma = close[i] < hma_4h_aligned[i]
        
        # --- 1d Volume Spike Confirmation ---
        volume_spike = volume[i] > 2.0 * vol_ma_1d_aligned[i]
        
        # --- Camarilla Level Conditions ---
        # Long: Price crosses above H3 with volume spike and bullish trend
        long_breakout = close[i] > camarilla_h3[i] and close[i-1] <= camarilla_h3[i-1]
        # Short: Price crosses below L3 with volume spike and bearish trend
        short_breakout = close[i] < camarilla_l3[i] and close[i-1] >= camarilla_l3[i-1]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Take profit at H4 level
                if close[i] >= camarilla_h4[i]:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.0 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Take profit at L4 level
                if close[i] <= camarilla_l4[i]:
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
        # Long: Camarilla H3 breakout up + volume spike + price above 4h HMA
        long_condition = long_breakout and volume_spike and price_above_hma
        
        # Short: Camarilla L3 breakout down + volume spike + price below 4h HMA
        short_condition = short_breakout and volume_spike and price_below_hma
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals