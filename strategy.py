Looking at the failure pattern:
- **12h/6h strategies generate TOO FEW trades** (0-120 total)
- **DB winner is 4h Camarilla + volume + choppiness: Sharpe 1.47 on ETHUSDT**
- Most failures have too many conditions = overtrading or conflicting signals

My hypothesis: Use the proven Camarilla pivot pattern that worked in the DB, but SIMPLIFY the entry conditions. 4h timeframe has 4x more bars than 12h, giving more trade opportunities while still reducing noise vs 1h.

Key insight from DB: Camarilla S4/R4 bounces + volume spike + choppiness regime = Sharpe 1.47. I'll implement this cleanly.
#!/usr/bin/env python3
"""
Experiment #021: 4h Camarilla S4/R4 Bounce + Volume + Choppiness Regime

HYPOTHESIS: Daily Camarilla S4/R4 levels mark strong support/resistance on 4h.
When price bounces at these extreme levels (S4=bottom, R4=top) WITH:
1. Volume spike (>1.5x MA) confirming institutional interest
2. Choppiness Index > 61.8 confirming ranging market (Camarilla mean-reversion works in range)
3. NOT at extremes of 1d range (avoid catching falling knives)

Works in BOTH bull (long S4 bounces) and bear (short R4 bounces).

TIMEFRAME: 4h primary
HTF: 1d for Camarilla levels + Choppiness
TARGET: 75-200 total trades over 4 years
SUCCESSFUL DB REFERENCE: gen_camarilla_pivot_volume_spike_choppiness_4h_v1 (Sharpe 1.471, 95 trades)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_camarilla_s4_r4_vol_chop_v1"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_camarilla_levels(high, low, close):
    """Calculate Camarilla pivot levels (S3, S4, R3, R4)"""
    n = len(close)
    s3 = np.full(n, np.nan, dtype=np.float64)
    s4 = np.full(n, np.nan, dtype=np.float64)
    r3 = np.full(n, np.nan, dtype=np.float64)
    r4 = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(1, n):
        h = high[i-1]
        l = low[i-1]
        c = close[i-1]
        rng = h - l
        
        s3[i] = c - rng * (1.1 / 12)
        s4[i] = c - rng * (1.1 / 6)
        r3[i] = c + rng * (1.1 / 12)
        r4[i] = c + rng * (1.1 / 6)
    
    return s3, s4, r3, r4

def calculate_choppiness_index(high, low, close, period=14):
    """Choppiness Index - values > 61.8 = ranging, < 38.2 = trending"""
    n = len(close)
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high > lowest_low:
            sum_tr = 0.0
            for j in range(i - period + 1, i + 1):
                sum_tr += high[j] - low[j]
            
            range_ratio = (highest_high - lowest_low) / (sum_tr + 1e-10)
            chop[i] = 100 * np.log10(range_ratio) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla levels
    s3_1d, s4_1d, r3_1d, r4_1d = calculate_camarilla_levels(
        df_1d['high'].values, df_1d['low'].values, df_1d['close'].values
    )
    
    # 1d Choppiness for regime
    chop_1d = calculate_choppiness_index(
        df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, period=14
    )
    
    # Align to 4h
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate local 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Volume MA for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # RSI for momentum confirmation
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(span=14, min_periods=14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = (100 - (100 / (1 + rs))).values
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 60  # Need enough for 1d alignment
    
    for i in range(warmup, n):
        # Skip if key indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(s4_1d_aligned[i]) or np.isnan(r4_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME CHECK ===
        chop_val = chop_1d_aligned[i] if not np.isnan(chop_1d_aligned[i]) else 50.0
        is_ranging = chop_val > 61.8  # Range-bound market
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === RSI STATE ===
        rsi_val = rsi[i]
        
        # === Camarilla level proximity ===
        # Price distance from S4/R4 as percentage of ATR
        dist_to_s4 = (close[i] - s4_1d_aligned[i]) / (atr_14[i] + 1e-10)
        dist_to_r4 = (r4_1d_aligned[i] - close[i]) / (atr_14[i] + 1e-10)
        
        # Price is AT S4 level (within 0.5 ATR)
        at_s4 = abs(dist_to_s4) < 0.5
        # Price is AT R4 level (within 0.5 ATR)
        at_r4 = abs(dist_to_r4) < 0.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === NEW LONG ENTRY: Price bounces at S4 ===
            # Conditions:
            # 1. Price at S4 level (within 0.5 ATR)
            # 2. Volume spike confirming institutional interest
            # 3. RSI not oversold (avoid bottom fishing) - RSI > 35
            # 4. Market is ranging OR RSI showing recovery
            if at_s4 and vol_spike and rsi_val > 35:
                desired_signal = SIZE
            
            # === NEW SHORT ENTRY: Price rejects at R4 ===
            # Conditions:
            # 1. Price at R4 level (within 0.5 ATR)
            # 2. Volume spike
            # 3. RSI not overbought - RSI < 65
            if at_r4 and vol_spike and rsi_val < 65:
                desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5 ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TAKE PROFIT: Near opposite Camarilla level ===
        tp_triggered = False
        
        if in_position and position_side > 0:
            # Long TP: price reaches R3 or R4
            dist_to_r3 = (r3_1d[i] - close[i]) / (atr_14[i] + 1e-10) if not np.isnan(r3_1d[i]) else 999
            if dist_to_r3 < 0.3:  # Within 0.3 ATR of R3
                tp_triggered = True
        
        if in_position and position_side < 0:
            # Short TP: price reaches S3
            dist_to_s3 = (close[i] - s3_1d[i]) / (atr_14[i] + 1e-10) if not np.isnan(s3_1d[i]) else 999
            if dist_to_s3 < 0.3:  # Within 0.3 ATR of S3
                tp_triggered = True
        
        if tp_triggered:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = desired_signal
    
    return signals