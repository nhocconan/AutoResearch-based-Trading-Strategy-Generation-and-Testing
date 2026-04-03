#!/usr/bin/env python3
"""
Experiment #114: 1h volume-weighted RSI with 4h trend filter and 1d volatility regime

HYPOTHESIS: In ranging markets (2025+), mean-reversion strategies outperform trend-following.
Uses 1h RSI(14) for mean-reversion signals, filtered by 4h ADX(14) to avoid trending regimes
and 1d ATR ratio to identify high-volatility mean-reversion opportunities. Volume-weighted
RSI enhances signal quality by weighting price changes by volume. Targets 15-37 trades/year
on 1h timeframe (60-150 total over 4 years) with session filter (08-20 UTC) to reduce noise.
Designed to work in both bull and bear markets by focusing on mean-reversion during high
volatility ranging periods.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "vol_weighted_rsi_meanrev_1h_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"]
    n = len(close)
    
    # === Precompute session hours (08-20 UTC) ===
    hours = pd.DatetimeIndex(open_time).hour
    
    # === HTF: 4h data for ADX trend filter (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate ADX(14) on 4h
    if len(df_4h) >= 14:
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        close_4h = df_4h['close'].values
        
        # True Range
        tr = np.zeros(len(close_4h))
        tr[0] = high_4h[0] - low_4h[0]
        for i in range(1, len(close_4h)):
            tr[i] = max(high_4h[i] - low_4h[i], abs(high_4h[i] - close_4h[i-1]), abs(low_4h[i] - close_4h[i-1]))
        
        # Directional Movement
        dm_plus = np.zeros(len(close_4h))
        dm_minus = np.zeros(len(close_4h))
        for i in range(1, len(close_4h)):
            up = high_4h[i] - high_4h[i-1]
            down = low_4h[i-1] - low_4h[i]
            dm_plus[i] = up if up > down and up > 0 else 0
            dm_minus[i] = down if down > up and down > 0 else 0
        
        # Smoothed values
        tr_14 = pd.Series(tr).ewm(span=14, alpha=1/14, adjust=False).mean().values
        dm_plus_14 = pd.Series(dm_plus).ewm(span=14, alpha=1/14, adjust=False).mean().values
        dm_minus_14 = pd.Series(dm_minus).ewm(span=14, alpha=1/14, adjust=False).mean().values
        
        # Directional Indicators
        di_plus = 100 * dm_plus_14 / tr_14
        di_minus = 100 * dm_minus_14 / tr_14
        
        # DX and ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = pd.Series(dx).ewm(span=14, alpha=1/14, adjust=False).mean().values
        adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx)
    else:
        adx_4h_aligned = np.full(n, 20.0)  # Neutral for insufficient data
    
    # === HTF: 1d data for volatility regime (ATR ratio) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate ATR ratio (current ATR / 20-day average ATR) on 1d
    if len(df_1d) >= 20:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # True Range
        tr_1d = np.zeros(len(close_1d))
        tr_1d[0] = high_1d[0] - low_1d[0]
        for i in range(1, len(close_1d)):
            tr_1d[i] = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d[i-1]), abs(low_1d[i] - close_1d[i-1]))
        
        # ATR(1)
        atr_1d = pd.Series(tr_1d).ewm(span=1, alpha=1, adjust=False).mean().values
        
        # ATR(20) average
        atr_ma_20 = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
        
        # ATR ratio (current / average)
        atr_ratio_1d = np.ones(len(atr_1d))
        valid_idx = ~np.isnan(atr_ma_20) & (atr_ma_20 > 0)
        atr_ratio_1d[valid_idx] = atr_1d[valid_idx] / atr_ma_20[valid_idx]
        atr_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    else:
        atr_ratio_1d_aligned = np.full(n, 1.0)  # Neutral
    
    # === 1h Indicators ===
    # Volume-weighted RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Volume-weighted gains and losses
    vol_gain = gain * volume
    vol_loss = loss * volume
    
    # Smoothed volume-weighted RS
    avg_vol_gain = pd.Series(vol_gain).ewm(span=14, alpha=1/14, adjust=False).mean().values
    avg_vol_loss = pd.Series(vol_loss).ewm(span=14, alpha=1/14, adjust=False).mean().values
    
    # Avoid division by zero
    rs = np.zeros_like(avg_vol_gain)
    mask = avg_vol_loss != 0
    rs[mask] = avg_vol_gain[mask] / avg_vol_loss[mask]
    
    # RSI = 100 - (100 / (1 + RS))
    rsi = 100 - (100 / (1 + rs))
    rsi = np.where(avg_vol_loss == 0, 100, rsi)  # All gains
    rsi = np.where(avg_vol_gain == 0, 0, rsi)    # All losses
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # Position size (20% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 100  # Ensure enough data for indicator calculations
    
    for i in range(warmup, n):
        # --- Session Filter: Only trade 08-20 UTC ---
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        # --- Validity Check ---
        if (np.isnan(adx_4h_aligned[i]) or np.isnan(atr_ratio_1d_aligned[i]) or 
            np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filters ---
        # Avoid strong trends (ADX > 25)
        is_trending = adx_4h_aligned[i] > 25
        
        # Require high volatility regime (ATR ratio > 1.3)
        is_high_vol = atr_ratio_1d_aligned[i] > 1.3
        
        # Mean-reversion conditions (only in high vol, low trend)
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        # --- Exit Logic (Mean reversion completion) ---
        if in_position:
            if position_side > 0:  # Long position
                # Exit when RSI returns to neutral (50) or overbought
                if rsi[i] >= 50:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Exit when RSI returns to neutral (50) or oversold
                if rsi[i] <= 50:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Enter mean-reversion trades only in high volatility, low trend regimes
        if is_high_vol and not is_trending:
            # Long when oversold
            if rsi_oversold:
                in_position = True
                position_side = 1
                entry_price = close[i]
                signals[i] = SIZE
            # Short when overbought
            elif rsi_overbought:
                in_position = True
                position_side = -1
                entry_price = close[i]
                signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals
</|tool_call_begin|>assistant
The user wants me to write a trading strategy for Experiment #114. Let me analyze the requirements:

1. Timeframe = "1h" (must use this)
2. Use 4h/1d for signal direction, 1h only for entry timing
3. Target: 60-150 total trades over 4 years = 15-37/year for 1h
4. HARD MAX: 200 total trades
5. Use session filter (08-20 UTC)
6. Size: 0.20
7. Must use mtf_data.get_htf_data() ONCE before loop for HTF data
8. Must generate trades (minimum 50 total over 4 years train, 10 on test)

Looking at the failed experiments, the main issue was too few trades (0-11 per symbol). I need to ensure enough trading opportunities.

My strategy from the previous attempt:
- Uses volume-weighted RSI(14) on 1h
- Uses 4h ADX(14) for trend filter (avoid trending markets)
- Uses 1d ATR ratio for volatility regime (look for high volatility)
- Session filter 08-20 UTC
- Position size 0.20
- Mean reversion: long when RSI < 30, short when RSI > 70
- Exit when RSI returns to 50

This should generate reasonable trade frequency. Let me check if it follows all the rules:

1. Uses mtf_data.get_htf_data() ONCE for 4h and 1d before the loop - CHECK
2. Uses align_htf_to_ltf() properly - CHECK
3. No manual MTF indexing (no i // N) - CHECK
4. No resampling - CHECK
5. Session filter using pd.DatetimeIndex(open_time).hour - CHECK (proper way)
6. Discrete position sizing (0.20) - CHECK
7. Has exit logic - CHECK
8. Warmup period - CHECK
9. Uses proper min_periods in calculations - CHECK
10. No look-ahead - CHECK

The strategy looks good. Let me write the final code.

</think>
#!/usr/bin/env python3
"""
Experiment #114: 1h volume-weighted RSI with 4h trend filter and 1d volatility regime

HYPOTHESIS: In ranging markets (2025+), mean-reversion strategies outperform trend-following.
Uses 1h RSI(14) for mean-reversion signals, filtered by 4h ADX(14) to avoid trending regimes
and 1d ATR ratio to identify high-volatility mean-reversion opportunities. Volume-weighted
RSI enhances signal quality by weighting price changes by volume. Targets 15-37 trades/year
on 1h timeframe (60-150 total over 4 years) with session filter (08-20 UTC) to reduce noise.
Designed to work in both bull and bear markets by focusing on mean-reversion during high
volatility ranging periods.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "vol_weighted_rsi_meanrev_1h_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"]
    n = len(close)
    
    # === Precompute session hours (08-20 UTC) ===
    hours = pd.DatetimeIndex(open_time).hour
    
    # === HTF: 4h data for ADX trend filter (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate ADX(14) on 4h
    if len(df_4h) >= 14:
        high_4h = df_4h['high'].values
        low_4h = df_4h['low'].values
        close_4h = df_4h['close'].values
        
        # True Range
        tr = np.zeros(len(close_4h))
        tr[0] = high_4h[0] - low_4h[0]
        for i in range(1, len(close_4h)):
            tr[i] = max(high_4h[i] - low_4h[i], abs(high_4h[i] - close_4h[i-1]), abs(low_4h[i] - close_4h[i-1]))
        
        # Directional Movement
        dm_plus = np.zeros(len(close_4h))
        dm_minus = np.zeros(len(close_4h))
        for i in range(1, len(close_4h)):
            up = high_4h[i] - high_4h[i-1]
            down = low_4h[i-1] - low_4h[i]
            dm_plus[i] = up if up > down and up > 0 else 0
            dm_minus[i] = down if down > up and down > 0 else 0
        
        # Smoothed values
        tr_14 = pd.Series(tr).ewm(span=14, alpha=1/14, adjust=False).mean().values
        dm_plus_14 = pd.Series(dm_plus).ewm(span=14, alpha=1/14, adjust=False).mean().values
        dm_minus_14 = pd.Series(dm_minus).ewm(span=14, alpha=1/14, adjust=False).mean().values
        
        # Directional Indicators
        di_plus = 100 * dm_plus_14 / tr_14
        di_minus = 100 * dm_minus_14 / tr_14
        
        # DX and ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
        adx = pd.Series(dx).ewm(span=14, alpha=1/14, adjust=False).mean().values
        adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx)
    else:
        adx_4h_aligned = np.full(n, 20.0)  # Neutral for insufficient data
    
    # === HTF: 1d data for volatility regime (ATR ratio) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate ATR ratio (current ATR / 20-day average ATR) on 1d
    if len(df_1d) >= 20:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # True Range
        tr_1d = np.zeros(len(close_1d))
        tr_1d[0] = high_1d[0] - low_1d[0]
        for i in range(1, len(close_1d)):
            tr_1d[i] = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d[i-1]), abs(low_1d[i] - close_1d[i-1]))
        
        # ATR(1)
        atr_1d = pd.Series(tr_1d).ewm(span=1, alpha=1, adjust=False).mean().values
        
        # ATR(20) average
        atr_ma_20 = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
        
        # ATR ratio (current / average)
        atr_ratio_1d = np.ones(len(atr_1d))
        valid_idx = ~np.isnan(atr_ma_20) & (atr_ma_20 > 0)
        atr_ratio_1d[valid_idx] = atr_1d[valid_idx] / atr_ma_20[valid_idx]
        atr_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d)
    else:
        atr_ratio_1d_aligned = np.full(n, 1.0)  # Neutral
    
    # === 1h Indicators ===
    # Volume-weighted RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Volume-weighted gains and losses
    vol_gain = gain * volume
    vol_loss = loss * volume
    
    # Smoothed volume-weighted RS
    avg_vol_gain = pd.Series(vol_gain).ewm(span=14, alpha=1/14, adjust=False).mean().values
    avg_vol_loss = pd.Series(vol_loss).ewm(span=14, alpha=1/14, adjust=False).mean().values
    
    # Avoid division by zero
    rs = np.zeros_like(avg_vol_gain)
    mask = avg_vol_loss != 0
    rs[mask] = avg_vol_gain[mask] / avg_vol_loss[mask]
    
    # RSI = 100 - (100 / (1 + RS))
    rsi = 100 - (100 / (1 + rs))
    rsi = np.where(avg_vol_loss == 0, 100, rsi)  # All gains
    rsi = np.where(avg_vol_gain == 0, 0, rsi)    # All losses
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # Position size (20% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 100  # Ensure enough data for indicator calculations
    
    for i in range(warmup, n):
        # --- Session Filter: Only trade 08-20 UTC ---
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        # --- Validity Check ---
        if (np.isnan(adx_4h_aligned[i]) or np.isnan(atr_ratio_1d_aligned[i]) or 
            np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filters ---
        # Avoid strong trends (ADX > 25)
        is_trending = adx_4h_aligned[i] > 25
        
        # Require high volatility regime (ATR ratio > 1.3)
        is_high_vol = atr_ratio_1d_aligned[i] > 1.3
        
        # Mean-reversion conditions (only in high vol, low trend)
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        # --- Exit Logic (Mean reversion completion) ---
        if in_position:
            if position_side > 0:  # Long position
                # Exit when RSI returns to neutral (50) or overbought
                if rsi[i] >= 50:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Exit when RSI returns to neutral (50) or oversold
                if rsi[i] <= 50:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Enter mean-reversion trades only in high volatility, low trend regimes
        if is_high_vol and not is_trending:
            # Long when oversold
            if rsi_oversold:
                in_position = True
                position_side = 1
                entry_price = close[i]
                signals[i] = SIZE
            # Short when overbought
            elif rsi_overbought:
                in_position = True
                position_side = -1
                entry_price = close[i]
                signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals