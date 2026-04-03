#!/usr/bin/env python3
"""
Experiment #034: 1h Camarilla Pivot + 4h/1d Trend Filter + Volume Spike

HYPOTHESIS: Camarilla pivot levels on 1h provide precise intraday support/resistance. 
Breakouts above R4 or below S4 with volume confirmation (>1.5x average) and alignment 
with 4h/1d trend (price > EMA50 on both) capture momentum moves. 
Using 4h/1d for trend direction and 1h only for entry timing reduces false signals. 
Session filter (08-20 UTC) avoids low-volume periods. Target: 15-37 trades/year (60-150 total over 4 years) 
to minimize fee drag while maintaining statistical significance.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_camarilla_pivot_vol_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    
    # === HTF: 4h and 1d data for trend filters (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA(50) on 4h close
    if len(df_4h) >= 50:
        close_4h = df_4h['close'].values
        ema_50_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    else:
        ema_50_4h_aligned = np.full(n, np.nan)
    
    # Calculate EMA(50) on 1d close
    if len(df_1d) >= 50:
        close_1d = df_1d['close'].values
        ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    else:
        ema_50_1d_aligned = np.full(n, np.nan)
    
    # === 1h Indicators: Camarilla Pivot Levels (based on previous day) ===
    # Calculate daily pivot from previous day's OHLC
    # We'll use rolling window of 24 bars (24h = 1d on 1h timeframe)
    roll_high_24 = pd.Series(high).rolling(window=24, min_periods=24).max().values
    roll_low_24 = pd.Series(low).rolling(window=24, min_periods=24).min().values
    roll_close_24 = pd.Series(close).rolling(window=24, min_periods=24).last().values
    
    # Camarilla levels: based on previous day's range
    rng = roll_high_24 - roll_low_24
    camarilla_h4 = roll_close_24 + (rng * 1.1 / 2)  # R4
    camarilla_l4 = roll_close_24 - (rng * 1.1 / 2)  # S4
    camarilla_h3 = roll_close_24 + (rng * 1.1 / 4)  # R3
    camarilla_l3 = roll_close_24 - (rng * 1.1 / 4)  # S3
    
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
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # Discrete position sizing (20% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 100  # Ensure enough data for indicators
    
    for i in range(warmup, n):
        # --- Session Filter: Only trade 08-20 UTC ---
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(camarilla_h4[i]) or np.isnan(camarilla_l4[i]) or 
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Trend Filter: Only trade in direction of BOTH 4h and 1d EMA50 ---
        price_above_both_emas = close[i] > ema_50_4h_aligned[i] and close[i] > ema_50_1d_aligned[i]
        price_below_both_emas = close[i] < ema_50_4h_aligned[i] and close[i] < ema_50_1d_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- Camarilla Breakout Conditions ---
        breakout_up = close[i] > camarilla_h4[i]
        breakout_down = close[i] < camarilla_l4[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at Camarilla H3/L3
                if close[i] > camarilla_h3[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.0 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at Camarilla L3/H3
                if close[i] < camarilla_l3[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Camarilla breakout up + volume spike + price above BOTH 4h and 1d EMA50
        long_condition = breakout_up and volume_spike and price_above_both_emas
        
        # Short: Camarilla breakout down + volume spike + price below BOTH 4h and 1d EMA50
        short_condition = breakout_down and volume_spike and price_below_both_emas
        
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
Experiment #034: 1h Camarilla Pivot + 4h/1d Trend Filter + Volume Spike

HYPOTHESIS: Camarilla pivot levels on 1h provide precise intraday support/resistance. 
Breakouts above R4 or below S4 with volume confirmation (>1.5x average) and alignment 
with 4h/1d trend (price > EMA50 on both) capture momentum moves. 
Using 4h/1d for trend direction and 1h only for entry timing reduces false signals. 
Session filter (08-20 UTC) avoids low-volume periods. Target: 15-37 trades/year (60-150 total over 4 years) 
to minimize fee drag while maintaining statistical significance.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_camarilla_pivot_vol_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    
    # === HTF: 4h and 1d data for trend filters (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA(50) on 4h close
    if len(df_4h) >= 50:
        close_4h = df_4h['close'].values
        ema_50_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    else:
        ema_50_4h_aligned = np.full(n, np.nan)
    
    # Calculate EMA(50) on 1d close
    if len(df_1d) >= 50:
        close_1d = df_1d['close'].values
        ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    else:
        ema_50_1d_aligned = np.full(n, np.nan)
    
    # === 1h Indicators: Camarilla Pivot Levels (based on previous day) ===
    # Calculate daily pivot from previous day's OHLC
    # We'll use rolling window of 24 bars (24h = 1d on 1h timeframe)
    roll_high_24 = pd.Series(high).rolling(window=24, min_periods=24).max().values
    roll_low_24 = pd.Series(low).rolling(window=24, min_periods=24).min().values
    roll_close_24 = pd.Series(close).rolling(window=24, min_periods=24).last().values
    
    # Camarilla levels: based on previous day's range
    rng = roll_high_24 - roll_low_24
    camarilla_h4 = roll_close_24 + (rng * 1.1 / 2)  # R4
    camarilla_l4 = roll_close_24 - (rng * 1.1 / 2)  # S4
    camarilla_h3 = roll_close_24 + (rng * 1.1 / 4)  # R3
    camarilla_l3 = roll_close_24 - (rng * 1.1 / 4)  # S3
    
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
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # Discrete position sizing (20% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 100  # Ensure enough data for indicators
    
    for i in range(warmup, n):
        # --- Session Filter: Only trade 08-20 UTC ---
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(camarilla_h4[i]) or np.isnan(camarilla_l4[i]) or 
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Trend Filter: Only trade in direction of BOTH 4h and 1d EMA50 ---
        price_above_both_emas = close[i] > ema_50_4h_aligned[i] and close[i] > ema_50_1d_aligned[i]
        price_below_both_emas = close[i] < ema_50_4h_aligned[i] and close[i] < ema_50_1d_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- Camarilla Breakout Conditions ---
        breakout_up = close[i] > camarilla_h4[i]
        breakout_down = close[i] < camarilla_l4[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at Camarilla H3/L3
                if close[i] > camarilla_h3[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.0 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at Camarilla L3/H3
                if close[i] < camarilla_l3[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Camarilla breakout up + volume spike + price above BOTH 4h and 1d EMA50
        long_condition = breakout_up and volume_spike and price_above_both_emas
        
        # Short: Camarilla breakout down + volume spike + price below BOTH 4h and 1d EMA50
        short_condition = breakout_down and volume_spike and price_below_both_emas
        
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