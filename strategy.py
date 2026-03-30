#!/usr/bin/env python3
"""
Experiment #028: 4h TRIX Crossover + Volume Spike + 1d Trend Filter

HYPOTHESIS: TRIX is a proven momentum oscillator that catches trend reversals early.
By requiring TRIX to cross its signal line (not just be above), we get stronger entries.
Combined with volume spike confirmation and 1d SMA200 trend filter, this should:
- Catch major trend changes early (TRIX crossover)
- Avoid false breakouts (volume confirmation)
- Stay aligned with longer-term trend (1d SMA200 filter)

WHY IT WORKS IN BULL AND BEAR:
- TRIX crossover works for both long and short entries
- 1d SMA200 filter identifies bull/bear regime
- Shorting in bear when TRIX crosses down + price below SMA200
- Going long in bull when TRIX crosses up + price above SMA200

TARGET: 75-200 total trades over 4 years (19-50/year). HARD MAX: 400.
Signal size: 0.30.

Based on: ETHUSDT TRIX test Sharpe 1.32, but with tighter entries to reduce trades.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_trix_vol_sma200_chop_v1"
timeframe = "4h"
leverage = 1.0

def calculate_trix(close, period=9):
    """TRIX - Triple EMA derivative - measures momentum"""
    n = len(close)
    if n < period * 3:
        return np.full(n, np.nan)
    
    # Triple EMA
    ema1 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    ema2 = pd.Series(ema1).ewm(span=period, min_periods=period, adjust=False).mean().values
    ema3 = pd.Series(ema2).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # TRIX = rate of change of triple EMA
    trix = np.full(n, np.nan, dtype=np.float64)
    for i in range(period * 3, n):
        if ema3[i - period] != 0:
            trix[i] = 100 * (ema3[i] - ema3[i - period]) / ema3[i - period]
    
    return trix

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

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - lower = trending, higher = choppy"""
    n = len(high)
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        tr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            if j > 0:
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]))
            else:
                tr = high[j] - low[j]
            tr_sum += tr
        
        if tr_sum > 0:
            hh = np.max(high[i - period + 1:i + 1])
            ll = np.min(low[i - period + 1:i + 1])
            range_hl = hh - ll
            
            if range_hl > 0:
                chop[i] = 100 * np.log10(tr_sum / range_hl) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d SMA200 for trend direction
    sma_1d = pd.Series(df_1d['close'].values).rolling(window=200, min_periods=200).mean().values
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d)
    
    # Local indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # TRIX (period 9) with signal line (period 9)
    trix = calculate_trix(close, period=9)
    trix_signal = pd.Series(trix).ewm(span=9, min_periods=9, adjust=False).mean().values
    
    # Volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
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
    entry_bar = 0
    trix_was_below = False
    trix_was_above = False
    
    warmup = 500  # Need enough for TRIX triple EMA + SMA200(1d)
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(sma_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(trix[i]) or np.isnan(trix_signal[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (1d SMA200) ===
        price_above_1d_sma = close[i] > sma_1d_aligned[i]
        
        # === REGIME (Choppiness Index) ===
        # Only trade when not too choppy (CHOP < 61.8)
        is_choppy = chop[i] > 61.8
        
        # Skip if too choppy (unless already in position)
        if is_choppy and not in_position:
            signals[i] = 0.0
            continue
        
        # === TRIX CROSSOVER DETECTION ===
        # Track previous TRIX vs signal relationship
        trix_above = trix[i] > trix_signal[i]
        prev_trix_above = trix[i - 1] > trix_signal[i - 1] if i > warmup else trix_above
        
        # Crossover UP: TRIX crossed above signal line
        trix_cross_up = (not prev_trix_above) and trix_above
        # Crossover DOWN: TRIX crossed below signal line
        trix_cross_down = prev_trix_above and (not trix_above)
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: TRIX crosses above signal + price above SMA200 + volume ===
            if trix_cross_up and price_above_1d_sma and vol_spike:
                desired_signal = SIZE
            
            # === SHORT: TRIX crosses below signal + price below SMA200 + volume ===
            if trix_cross_down and not price_above_1d_sma and vol_spike:
                desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5 ATR trailing) ===
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
        
        # === TIME-BASED EXIT (hold at least 8 bars = 2 days) ===
        bars_held = i - entry_bar
        
        if in_position and bars_held >= 8:
            # Exit on opposite TRIX crossover
            if position_side > 0 and trix_cross_down:
                desired_signal = 0.0
            if position_side < 0 and trix_cross_up:
                desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
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