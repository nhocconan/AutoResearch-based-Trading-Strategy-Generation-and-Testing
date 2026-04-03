#!/usr/bin/env python3
"""
Experiment #055: 6h Donchian(20) breakout + 1d Camarilla pivot + volume confirmation
HYPOTHESIS: 6h Donchian breakouts aligned with 1d Camarilla pivot levels (R3/S3 for fade, R4/S4 for continuation) 
capture institutional order flow with volume confirmation. Weekly trend filter ensures alignment with higher timeframe momentum. 
Targets 75-150 trades over 4 years by requiring confluence of breakout, pivot level, volume spike, and weekly trend.
Works in bull/bear via weekly trend filter and pivot-based mean reversion/continuation logic.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_055_6h_donchian20_1d_camarilla_vol_v1"
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
    
    # Calculate Camarilla pivot levels for 1d
    def calculate_camarilla(high, low, close):
        # Typical price for pivot
        pp = (high + low + close) / 3.0
        range_hl = high - low
        
        # Camarilla levels
        r4 = pp + (range_hl * 1.1 / 2)
        r3 = pp + (range_hl * 1.1 / 4)
        r2 = pp + (range_hl * 1.1 / 6)
        r1 = pp + (range_hl * 1.1 / 12)
        s1 = pp - (range_hl * 1.1 / 12)
        s2 = pp - (range_hl * 1.1 / 6)
        s3 = pp - (range_hl * 1.1 / 4)
        s4 = pp - (range_hl * 1.1 / 2)
        
        return pp, r1, r2, r3, r4, s1, s2, s3, s4
    
    # Calculate for each 1d bar
    camarilla_pp = np.full(len(df_1d), np.nan)
    camarilla_r3 = np.full(len(df_1d), np.nan)
    camarilla_s3 = np.full(len(df_1d), np.nan)
    camarilla_r4 = np.full(len(df_1d), np.nan)
    camarilla_s4 = np.full(len(df_1d), np.nan)
    
    for i in range(len(df_1d)):
        pp, r1, r2, r3, r4, s1, s2, s3, s4 = calculate_camarilla(
            df_1d['high'].iloc[i], 
            df_1d['low'].iloc[i], 
            df_1d['close'].iloc[i]
        )
        camarilla_pp[i] = pp
        camarilla_r3[i] = r3
        camarilla_s3[i] = s3
        camarilla_r4[i] = r4
        camarilla_s4[i] = s4
    
    # Align to 6h timeframe
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # === HTF: 1w data for weekly trend filter ===
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA(21) for trend
    ema_21_1w = pd.Series(df_1w['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # === 6h Indicators: Donchian(20) channels ===
    def calculate_donchian(high, low, period=20):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donch_upper, donch_lower = calculate_donchian(high, low, 20)
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr_6h = np.zeros(n)
    tr_6h[0] = high[0] - low[0]
    for i in range(1, n):
        tr_6h[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr_6h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0  # Track bars in position for minimum holding period
    
    warmup = 50  # Warmup for indicator stability
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(camarilla_pp_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or np.isnan(ema_21_1w_aligned[i]) or
            np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or np.isnan(atr_14[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # === Weekly Trend Filter ===
        weekly_uptrend = price > ema_21_1w_aligned[i]
        weekly_downtrend = price < ema_21_1w_aligned[i]
        
        # === Camarilla Logic ===
        # Fade at R3/S3 (mean reversion)
        fade_long = price <= camarilla_s3_aligned[i] and price >= camarilla_s4_aligned[i]
        fade_short = price >= camarilla_r3_aligned[i] and price <= camarilla_r4_aligned[i]
        
        # Breakout continuation at R4/S4
        breakout_long = price >= camarilla_r4_aligned[i]
        breakout_short = price <= camarilla_s4_aligned[i]
        
        # === 6h Donchian Breakout ===
        donch_breakout_up = high[i] > donch_upper[i-1]
        donch_breakout_down = low[i] < donch_lower[i-1]
        
        # === Volume Confirmation ===
        volume_spike = vol_ratio[i] > 2.0
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit on Donchian breakdown with volume (profit taking/stop)
                if donch_breakout_down and volume_spike:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit on Donchian breakout up with volume (profit taking/stop)
                if donch_breakout_up and volume_spike:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 3 bars to reduce churn
            if bars_since_entry < 3:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long logic: 
        # 1. Weekly uptrend AND Donchian breakout up AND volume spike (continuation)
        # 2. OR Weekly downtrend AND price at S3/S4 AND Donchian breakout up AND volume spike (fade)
        long_signal = False
        if weekly_uptrend and donch_breakout_up and volume_spike:
            long_signal = True  # Continuation in uptrend
        elif weekly_downtrend and fade_long and donch_breakout_up and volume_spike:
            long_signal = True  # Fade at support in downtrend
        
        # Short logic:
        # 1. Weekly downtrend AND Donchian breakout down AND volume spike (continuation)
        # 2. OR Weekly uptrend AND price at R3/R4 AND Donchian breakout down AND volume spike (fade)
        short_signal = False
        if weekly_downtrend and donch_breakout_down and volume_spike:
            short_signal = True  # Continuation in downtrend
        elif weekly_uptrend and fade_short and donch_breakout_down and volume_spike:
            short_signal = True  # Fade at resistance in uptrend
        
        # Execute signals
        if long_signal:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        elif short_signal:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals

</think>
#!/usr/bin/env python3
"""
Experiment #055: 6h Donchian(20) breakout + 1d Camarilla pivot + volume confirmation
HYPOTHESIS: 6h Donchian breakouts aligned with 1d Camarilla pivot levels (R3/S3 for fade, R4/S4 for continuation) 
capture institutional order flow with volume confirmation. Weekly trend filter ensures alignment with higher timeframe momentum. 
Targets 75-150 trades over 4 years by requiring confluence of breakout, pivot level, volume spike, and weekly trend.
Works in bull/bear via weekly trend filter and pivot-based mean reversion/continuation logic.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_055_6h_donchian20_1d_camarilla_vol_v1"
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
    
    # Calculate Camarilla pivot levels for 1d
    def calculate_camarilla(high, low, close):
        # Typical price for pivot
        pp = (high + low + close) / 3.0
        range_hl = high - low
        
        # Camarilla levels
        r4 = pp + (range_hl * 1.1 / 2)
        r3 = pp + (range_hl * 1.1 / 4)
        r2 = pp + (range_hl * 1.1 / 6)
        r1 = pp + (range_hl * 1.1 / 12)
        s1 = pp - (range_hl * 1.1 / 12)
        s2 = pp - (range_hl * 1.1 / 6)
        s3 = pp - (range_hl * 1.1 / 4)
        s4 = pp - (range_hl * 1.1 / 2)
        
        return pp, r1, r2, r3, r4, s1, s2, s3, s4
    
    # Calculate for each 1d bar
    camarilla_pp = np.full(len(df_1d), np.nan)
    camarilla_r3 = np.full(len(df_1d), np.nan)
    camarilla_s3 = np.full(len(df_1d), np.nan)
    camarilla_r4 = np.full(len(df_1d), np.nan)
    camarilla_s4 = np.full(len(df_1d), np.nan)
    
    for i in range(len(df_1d)):
        pp, r1, r2, r3, r4, s1, s2, s3, s4 = calculate_camarilla(
            df_1d['high'].iloc[i], 
            df_1d['low'].iloc[i], 
            df_1d['close'].iloc[i]
        )
        camarilla_pp[i] = pp
        camarilla_r3[i] = r3
        camarilla_s3[i] = s3
        camarilla_r4[i] = r4
        camarilla_s4[i] = s4
    
    # Align to 6h timeframe
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # === HTF: 1w data for weekly trend filter ===
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA(21) for trend
    ema_21_1w = pd.Series(df_1w['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # === 6h Indicators: Donchian(20) channels ===
    def calculate_donchian(high, low, period=20):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donch_upper, donch_lower = calculate_donchian(high, low, 20)
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr_6h = np.zeros(n)
    tr_6h[0] = high[0] - low[0]
    for i in range(1, n):
        tr_6h[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr_6h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0  # Track bars in position for minimum holding period
    
    warmup = 50  # Warmup for indicator stability
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(camarilla_pp_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or np.isnan(ema_21_1w_aligned[i]) or
            np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or np.isnan(atr_14[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # === Weekly Trend Filter ===
        weekly_uptrend = price > ema_21_1w_aligned[i]
        weekly_downtrend = price < ema_21_1w_aligned[i]
        
        # === Camarilla Logic ===
        # Fade at R3/S3 (mean reversion)
        fade_long = price <= camarilla_s3_aligned[i] and price >= camarilla_s4_aligned[i]
        fade_short = price >= camarilla_r3_aligned[i] and price <= camarilla_r4_aligned[i]
        
        # Breakout continuation at R4/S4
        breakout_long = price >= camarilla_r4_aligned[i]
        breakout_short = price <= camarilla_s4_aligned[i]
        
        # === 6h Donchian Breakout ===
        donch_breakout_up = high[i] > donch_upper[i-1]
        donch_breakout_down = low[i] < donch_lower[i-1]
        
        # === Volume Confirmation ===
        volume_spike = vol_ratio[i] > 2.0
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit on Donchian breakdown with volume (profit taking/stop)
                if donch_breakout_down and volume_spike:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit on Donchian breakout up with volume (profit taking/stop)
                if donch_breakout_up and volume_spike:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 3 bars to reduce churn
            if bars_since_entry < 3:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long logic: 
        # 1. Weekly uptrend AND Donchian breakout up AND volume spike (continuation)
        # 2. OR Weekly downtrend AND price at S3/S4 AND Donchian breakout up AND volume spike (fade)
        long_signal = False
        if weekly_uptrend and donch_breakout_up and volume_spike:
            long_signal = True  # Continuation in uptrend
        elif weekly_downtrend and fade_long and donch_breakout_up and volume_spike:
            long_signal = True  # Fade at support in downtrend
        
        # Short logic:
        # 1. Weekly downtrend AND Donchian breakout down AND volume spike (continuation)
        # 2. OR Weekly uptrend AND price at R3/R4 AND Donchian breakout down AND volume spike (fade)
        short_signal = False
        if weekly_downtrend and donch_breakout_down and volume_spike:
            short_signal = True  # Continuation in downtrend
        elif weekly_uptrend and fade_short and donch_breakout_down and volume_spike:
            short_signal = True  # Fade at resistance in uptrend
        
        # Execute signals
        if long_signal:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        elif short_signal:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals