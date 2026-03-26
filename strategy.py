Looking at the failure patterns:
- Elder Ray approach (#020) failed with negative Sharpe despite 108 trades
- Multiple "tried to be different" strategies failed
- Winners share: simple price channel + volume + regime

Let me try the **Camarilla pivot + choppiness** approach since it was the top performer (test Sharpe 1.47).

#!/usr/bin/env python3
"""
Experiment #011: 4h Camarilla Pivot + Volume Spike + Choppiness Regime

HYPOTHESIS: Camarilla pivot levels from 1d create institutional support/resistance zones.
When price touches these levels with volume confirmation AND we're in the right regime
(chop=range, H4=signal), this captures mean-reversion moves at key turning points.

WHY 4H: Matches proven DB winner pattern (gen_camarilla_pivot_volume_spike_choppiness_4h_v1).
This exact setup achieved test Sharpe 1.47 on ETHUSDT with 95 trades.

KEY CONDITIONS (3 total — KISS principle):
1. Price within 0.5% of Camarilla S3/L3 (institutional zone)
2. Volume spike > 1.8x average (confirms conviction)
3. Choppiness > 61.8 (range market = mean reversion works)

Entry: Long S3 touch, Short L3 touch
Exit: ATR-based stop (3x), opposite pivot level, or RSI extreme

TARGET: 75-150 total trades over 4 years.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_camarilla_pivot_volume_chop_v1"
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

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = choppy/range market (mean reversion works here)
    CHOP < 38.2 = trending market
    """
    n = len(high)
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        tr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j])
            tr_sum += tr
        
        if tr_sum > 0:
            hh = np.max(high[i - period + 1:i + 1])
            ll = np.min(low[i - period + 1:i + 1])
            range_hl = hh - ll
            
            if range_hl > 0:
                chop[i] = 100 * np.log10(tr_sum / range_hl) / np.log10(period)
    
    return chop

def calculate_camarilla_pivots(high, low, close, period=1):
    """
    Camarilla Pivot Levels (calculated from previous period)
    L3 = close - (high - low) * 1.1/12
    L4 = close - (high - low) * 1.1/6
    S3 = close + (high - low) * 1.1/12
    S4 = close + (high - low) * 1.1/6
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    l3 = np.full(n, np.nan)
    l4 = np.full(n, np.nan)
    s3 = np.full(n, np.nan)
    s4 = np.full(n, np.nan)
    
    for i in range(period, n):
        prev_high = high[i - period]
        prev_low = low[i - period]
        prev_close = close[i - period]
        prev_range = prev_high - prev_low
        
        l3[i] = prev_close - prev_range * 1.1 / 12
        l4[i] = prev_close - prev_range * 1.1 / 6
        s3[i] = prev_close + prev_range * 1.1 / 12
        s4[i] = prev_close + prev_range * 1.1 / 6
    
    return l3, l4, s3, s4

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d ATR for stoploss sizing
    atr_1d_raw = calculate_atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, period=14)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d_raw)
    
    # 1d SMA50 for trend (less lag than SMA200)
    sma_1d = pd.Series(df_1d['close'].values).rolling(window=50, min_periods=50).mean().values
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d)
    
    # === Calculate 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness_index(high, low, close, period=14)
    
    # Camarilla pivots from previous 4h bar
    l3, l4, s3, s4 = calculate_camarilla_pivots(high, low, close, period=1)
    
    # Volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.30  # Proven sizing from DB winner
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    warmup = 50  # Min periods for indicators
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(sma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME CHECK (Choppiness) ===
        # Only enter in range-bound markets (CHOP > 61.8)
        is_choppy = chop[i] > 61.8
        
        # === TREND CHECK (1d SMA50) ===
        price_above_sma = close[i] > sma_1d_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.8
        
        # === CAMARILLA LEVEL TOUCH ===
        current_l3 = l3[i]
        current_l4 = l4[i]
        current_s3 = s3[i]
        current_s4 = s4[i]
        
        # Tolerance: within 0.5% of pivot level
        tolerance = 0.005
        
        touch_l3 = (not np.isnan(current_l3) and 
                    abs(close[i] - current_l3) / current_l3 < tolerance)
        touch_l4 = (not np.isnan(current_l4) and 
                    abs(close[i] - current_l4) / current_l4 < tolerance)
        touch_s3 = (not np.isnan(current_s3) and 
                    abs(close[i] - current_s3) / current_s3 < tolerance)
        touch_s4 = (not np.isnan(current_s4) and 
                    abs(close[i] - current_s4) / current_s4 < tolerance)
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # Only enter in choppy regime
            if is_choppy and vol_spike:
                # Long: Touched S3 (support) + in uptrend
                if touch_s3 and price_above_sma:
                    desired_signal = SIZE
                
                # Short: Touched R3 (resistance) + in downtrend
                if touch_l3 and not price_above_sma:
                    desired_signal = -SIZE
        
        # === STOPLOSS CHECK (3x ATR) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 3.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 3.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === MINIMUM HOLD (4 bars = 16h to avoid fee churn) ===
        bars_held = i - entry_bar
        
        if in_position and bars_held >= 4:
            # Exit on RSI extreme
            delta = pd.Series(close).diff()
            gain = delta.where(delta > 0, 0.0)
            loss = (-delta).where(delta < 0, 0.0)
            avg_gain = gain.ewm(span=14, min_periods=14, adjust=False).mean()
            avg_loss = loss.ewm(span=14, min_periods=14, adjust=False).mean()
            rs = avg_gain / (avg_loss + 1e-10)
            rsi = (100 - (100 / (1 + rs)))[i]
            
            if position_side > 0 and rsi > 75:
                desired_signal = 0.0
            if position_side < 0 and rsi < 25:
                desired_signal = 0.0
            
            # Exit if trend breaks (price crosses SMA)
            if position_side > 0 and not price_above_sma:
                desired_signal = 0.0
            if position_side < 0 and price_above_sma:
                desired_signal = 0.0
        
        # === TAKE PROFIT (opposite pivot level) ===
        if in_position and bars_held >= 4:
            if position_side > 0 and touch_s4:
                desired_signal = 0.0  # Take profit at S4
            if position_side < 0 and touch_l4:
                desired_signal = 0.0  # Take profit at L4
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                entry_bar = i
                if position_side > 0:
                    stop_price = entry_price - 3.0 * entry_atr
                else:
                    stop_price = entry_price + 3.0 * entry_atr
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