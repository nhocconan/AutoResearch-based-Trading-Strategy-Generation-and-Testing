#!/usr/bin/env python3
"""
Experiment #059: 6h Camarilla Pivot + Volume Spike + 12h Trend Filter

HYPOTHESIS: Camarilla pivot levels (calculated from 1d OHLC) act as strong intraday support/resistance.
Price breaking above R4 or below S4 with volume confirmation (>2.0x average) and alignment with 
12h trend (price > EMA50 for longs, price < EMA50 for shorts) captures strong momentum breaks.
The 6h timeframe targets 12-37 trades/year (50-150 total over 4 years), minimizing fee drag while 
allowing for meaningful statistical sampling. Volume confirmation filters low-conviction breaks.
ATR-based stoploss (2.5x ATR) manages risk. Works in both bull (breakouts continue trend) and 
bear (breakdowns accelerate downtrend) markets by requiring trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_059_6h_camarilla_12h_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivot calculation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous 1d bar (HLC of previous day)
    camarilla_h4 = np.full(n, np.nan)
    camarilla_l4 = np.full(n, np.nan)
    camarilla_h3 = np.full(n, np.nan)
    camarilla_l3 = np.full(n, np.nan)
    
    if len(df_1d) >= 2:
        # Use previous day's OHLC to calculate today's Camarilla levels
        # Shift by 1 to avoid look-ahead (use previous completed day)
        prev_high = df_1d['high'].shift(1).values
        prev_low = df_1d['low'].shift(1).values
        prev_close = df_1d['close'].shift(1).values
        prev_range = prev_high - prev_low
        
        # Camarilla formulas
        camarilla_h4_raw = prev_close + prev_range * 1.1 / 2
        camarilla_l4_raw = prev_close - prev_range * 1.1 / 2
        camarilla_h3_raw = prev_close + prev_range * 1.1 / 4
        camarilla_l3_raw = prev_close - prev_range * 1.1 / 4
        
        # Align to 6h timeframe (shift(1) already applied above for data integrity)
        camarilla_h4 = align_htf_to_ltf(prices, df_1d, camarilla_h4_raw)
        camarilla_l4 = align_htf_to_ltf(prices, df_1d, camarilla_l4_raw)
        camarilla_h3 = align_htf_to_ltf(prices, df_1d, camarilla_h3_raw)
        camarilla_l3 = align_htf_to_ltf(prices, df_1d, camarilla_l3_raw)
    else:
        camarilla_h4 = camarilla_l4 = camarilla_h3 = camarilla_l3 = np.full(n, np.nan)
    
    # === HTF: 12h data for trend filter (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate EMA(50) on 12h close
    if len(df_12h) >= 50:
        close_12h = df_12h['close'].values
        ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    else:
        ema_50_12h_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 50  # Ensure enough data for HTF EMA50 and ATR
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(camarilla_h4[i]) or np.isnan(camarilla_l4[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(atr_14[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Trend Filter: Only trade when price above 12h EMA50 (long) or below (short) ---
        price_above_12h_ema = close[i] > ema_50_12h_aligned[i]
        price_below_12h_ema = close[i] < ema_50_12h_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
        # --- Camarilla Breakout Conditions ---
        breakout_up = close[i] > camarilla_h4[i]
        breakout_down = close[i] < camarilla_l4[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Camarilla breakout up (above H4) + volume spike + price above 12h EMA50
        long_condition = breakout_up and volume_spike and price_above_12h_ema
        
        # Short: Camarilla breakout down (below L4) + volume spike + price below 12h EMA50
        short_condition = breakout_down and volume_spike and price_below_12h_ema
        
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

</think>
#!/usr/bin/env python3
"""
Experiment #059: 6h Camarilla Pivot + Volume Spike + 12h Trend Filter

HYPOTHESIS: Camarilla pivot levels (calculated from 1d OHLC) act as strong intraday support/resistance.
Price breaking above R4 or below S4 with volume confirmation (>2.0x average) and alignment with 
12h trend (price > EMA50 for longs, price < EMA50 for shorts) captures strong momentum breaks.
The 6h timeframe targets 12-37 trades/year (50-150 total over 4 years), minimizing fee drag while 
allowing for meaningful statistical sampling. Volume confirmation filters low-conviction breaks.
ATR-based stoploss (2.5x ATR) manages risk. Works in both bull (breakouts continue trend) and 
bear (breakdowns accelerate downtrend) markets by requiring trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_059_6h_camarilla_12h_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivot calculation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous 1d bar (HLC of previous day)
    camarilla_h4 = np.full(n, np.nan)
    camarilla_l4 = np.full(n, np.nan)
    camarilla_h3 = np.full(n, np.nan)
    camarilla_l3 = np.full(n, np.nan)
    
    if len(df_1d) >= 2:
        # Use previous day's OHLC to calculate today's Camarilla levels
        # Shift by 1 to avoid look-ahead (use previous completed day)
        prev_high = df_1d['high'].shift(1).values
        prev_low = df_1d['low'].shift(1).values
        prev_close = df_1d['close'].shift(1).values
        prev_range = prev_high - prev_low
        
        # Camarilla formulas
        camarilla_h4_raw = prev_close + prev_range * 1.1 / 2
        camarilla_l4_raw = prev_close - prev_range * 1.1 / 2
        camarilla_h3_raw = prev_close + prev_range * 1.1 / 4
        camarilla_l3_raw = prev_close - prev_range * 1.1 / 4
        
        # Align to 6h timeframe (shift(1) already applied above for data integrity)
        camarilla_h4 = align_htf_to_ltf(prices, df_1d, camarilla_h4_raw)
        camarilla_l4 = align_htf_to_ltf(prices, df_1d, camarilla_l4_raw)
        camarilla_h3 = align_htf_to_ltf(prices, df_1d, camarilla_h3_raw)
        camarilla_l3 = align_htf_to_ltf(prices, df_1d, camarilla_l3_raw)
    else:
        camarilla_h4 = camarilla_l4 = camarilla_h3 = camarilla_l3 = np.full(n, np.nan)
    
    # === HTF: 12h data for trend filter (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate EMA(50) on 12h close
    if len(df_12h) >= 50:
        close_12h = df_12h['close'].values
        ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    else:
        ema_50_12h_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 50  # Ensure enough data for HTF EMA50 and ATR
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(camarilla_h4[i]) or np.isnan(camarilla_l4[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(atr_14[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Trend Filter: Only trade when price above 12h EMA50 (long) or below (short) ---
        price_above_12h_ema = close[i] > ema_50_12h_aligned[i]
        price_below_12h_ema = close[i] < ema_50_12h_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
        # --- Camarilla Breakout Conditions ---
        breakout_up = close[i] > camarilla_h4[i]
        breakout_down = close[i] < camarilla_l4[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Camarilla breakout up (above H4) + volume spike + price above 12h EMA50
        long_condition = breakout_up and volume_spike and price_above_12h_ema
        
        # Short: Camarilla breakout down (below L4) + volume spike + price below 12h EMA50
        short_condition = breakout_down and volume_spike and price_below_12h_ema
        
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