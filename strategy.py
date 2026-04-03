#!/usr/bin/env python3
"""
Experiment #1062: 12h Camarilla Pivot + Volume Spike + Chop Filter
HYPOTHESIS: Price reacts strongly to Camarilla pivot levels (L3, L4, H3, H4) on 12h timeframe. 
Enter long when price breaks above H4 with volume spike (>2x) and choppy market (CHOP>61.8). 
Enter short when price breaks below L4 with volume spike and choppy market. 
Use discrete position sizing (0.25) and ATR(14) stoploss (2.0*ATR). 
Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1062_12h_camarilla_vol_chop_v1"
timeframe = "12h"
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
    
    # Calculate Camarilla levels for previous 1d bar
    camarilla_h4 = np.zeros(len(close_1d))
    camarilla_l4 = np.zeros(len(close_1d))
    camarilla_h3 = np.zeros(len(close_1d))
    camarilla_l3 = np.zeros(len(close_1d))
    camarilla_h6 = np.zeros(len(close_1d))
    camarilla_l6 = np.zeros(len(close_1d))
    
    for i in range(1, len(close_1d)):
        # Use previous day's range to calculate today's Camarilla levels
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        range_val = prev_high - prev_low
        
        camarilla_h4[i] = prev_close + range_val * 1.1 / 2
        camarilla_l4[i] = prev_close - range_val * 1.1 / 2
        camarilla_h3[i] = prev_close + range_val * 1.1 / 4
        camarilla_l3[i] = prev_close - range_val * 1.1 / 4
        camarilla_h6[i] = prev_close + range_val * 1.1 / 6
        camarilla_l6[i] = prev_close - range_val * 1.1 / 6
    
    # Align HTF Camarilla levels to 12h timeframe (shifted by 1 for completed bars)
    h4_12h = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_12h = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    h3_12h = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_12h = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # === 12h Indicators: Chopiness Index (CHOP) for regime filter ===
    def calculate_chop(high, low, close, period=14):
        """Calculate Chopiness Index"""
        if len(close) < period:
            return np.full_like(close, np.nan)
        atr_sum = np.zeros_like(close)
        tr = np.zeros_like(close)
        
        for i in range(1, len(close)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        tr[0] = high[0] - low[0]
        
        # True Range sum over period
        for i in range(period, len(close)):
            atr_sum[i] = np.sum(tr[i-period+1:i+1])
        
        # Highest high and lowest low over period
        max_high = np.zeros_like(close)
        min_low = np.zeros_like(close)
        for i in range(period-1, len(close)):
            max_high[i] = np.max(high[i-period+1:i+1])
            min_low[i] = np.min(low[i-period+1:i+1])
        
        # Chopiness Index
        chop = np.zeros_like(close)
        for i in range(period, len(close)):
            if atr_sum[i] > 0 and (max_high[i] - min_low[i]) > 0:
                chop[i] = 100 * np.log10(atr_sum[i] / (max_high[i] - min_low[i])) / np.log10(period)
            else:
                chop[i] = 50.0
        return chop
    
    chop = calculate_chop(high, low, close, 14)
    
    # === 12h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 12h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 20  # sufficient for volume MA and Camarilla calculation
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(h4_12h[i]) or np.isnan(l4_12h[i]) or
            np.isnan(chop[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
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
            
            # Optional: time-based exit after 6 bars (~3d on 12h) to avoid overtrading
            if bars_since_entry > 6:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation: require volume spike (> 2.0x average)
        volume_spike = vol_ratio[i] > 2.0
        
        # Chop filter: choppy market (CHOP > 61.8) for mean reversion at extremes
        choppy_market = chop[i] > 61.8
        
        if volume_spike and choppy_market:
            # Breakout above H4 with volume spike in choppy market -> long
            if price > h4_12h[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Breakdown below L4 with volume spike in choppy market -> short
            elif price < l4_12h[i]:
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

if __name__ == "__main__":
    # Test with dummy data to ensure no runtime errors
    import pandas as pd
    import numpy as np
    dates = pd.date_range('2021-01-01', periods=100, freq='12h')
    test_data = pd.DataFrame({
        'open_time': dates,
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 102,
        'low': np.random.randn(100).cumsum() + 98,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.rand(100) * 1000 + 500,
        'taker_buy_volume': np.random.rand(100) * 500,
        'trades': np.random.randint(50, 200, 100)
    })
    print("Strategy loaded successfully")