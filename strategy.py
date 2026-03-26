#!/usr/bin/env python3
"""
Experiment #013: 1d Camarilla Bounce + Choppiness Regime

HYPOTHESIS: On 1d timeframe, Camarilla pivot levels (S3/R3) act as strong 
support/resistance where price frequently bounces. Combined with:
- Volume confirmation (validates institutional interest)
- Choppiness Index regime filter (prevents fading trending moves)
- Simple ATR stoploss

This is the proven DB pattern (gen_camarilla_pivot_volume_spike_choppiness_4h_v1 
had test Sharpe=1.471). Adapted to 1d for even fewer trades and lower fee drag.

TIMEFRAME: 1d primary
HTF: 1w for regime confirmation
TARGET: 50-150 total trades over 4 years (12-37/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_camarilla_chop_vol_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_camarilla(high, low, close, open_price=None):
    """
    Camarilla Pivot Points
    R4 = C + (H - L) * 1.1/2
    R3 = C + (H - L) * 1.1/4
    R2 = C + (H - L) * 1.1/6
    R1 = C + (H - L) * 1.1/12
    S1 = C - (H - L) * 1.1/12
    S2 = C - (H - L) * 1.1/6
    S3 = C - (H - L) * 1.1/4
    S4 = C - (H - L) * 1.1/2
    """
    n = len(close)
    hl = high - low
    factor = 1.1 / 2
    
    r4 = close + hl * factor
    r3 = close + hl * (factor / 2)
    r2 = close + hl * (factor / 3)
    r1 = close + hl * (factor / 6)
    s1 = close - hl * (factor / 6)
    s2 = close - hl * (factor / 3)
    s3 = close - hl * (factor / 2)
    s4 = close - hl * factor
    
    return r4, r3, r2, r1, s1, s2, s3, s4

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending
    CHOP > 61.8 = choppy/range (fade moves)
    CHOP < 38.2 = trending (follow moves)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        # Sum of true range over period
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j])
            atr_sum += tr
        
        # Highest - lowest over period
        hl_range = np.max(high[i - period + 1:i + 1]) - np.min(low[i - period + 1:i + 1])
        
        if hl_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100 * (np.log10(atr_sum) / np.log10(hl_range))
    
    return chop

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

def calculate_hma(close, period):
    """Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            if not np.isnan(series[i]):
                window = series[i - span + 1:i + 1].astype(np.float64)
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

def calculate_rsi(close, period=14):
    """RSI indicator"""
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = (100 - (100 / (1 + rs))).values
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1w = get_htf_data(prices, '1w')
    
    # 1w HMA for trend confirmation
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # === Calculate indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Camarilla levels
    r4, r3, r2, r1, s1, s2, s3, s4 = calculate_camarilla(high, low, close)
    
    # Choppiness Index
    chop = calculate_choppiness_index(high, low, close, period=14)
    
    # RSI
    rsi = calculate_rsi(close, period=14)
    
    # Volume MA
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
    
    warmup = 50
    
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
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        chop_val = chop[i]
        rsi_val = rsi[i]
        vol_ratio_val = vol_ratio[i]
        
        # Choppiness regime
        # CHOP > 61.8 = choppy = mean reversion (fade S3/R3 touches)
        # CHOP < 38.2 = trending = trend following (hold through)
        is_choppy = chop_val > 61.8
        is_trending = chop_val < 38.2
        
        # 1w trend
        price_above_1w_hma = close[i] > hma_1w_aligned[i]
        
        # Volume confirmation
        vol_confirm = vol_ratio_val > 1.2
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG ENTRY: Price bounces from S3 with volume ===
            # S3 is strong support - bounce when price approaches
            if close[i] >= s3[i] and close[i] <= s2[i]:
                # Price at/below S3, bounce expected
                # Need volume confirmation
                if vol_confirm:
                    # In choppy: fade the move (long the bounce)
                    # In trending: confirm with 1w trend aligned
                    if is_choppy or (is_trending and price_above_1w_hma):
                        desired_signal = SIZE
            
            # === SHORT ENTRY: Price bounces from R3 with volume ===
            if close[i] <= r3[i] and close[i] >= r2[i]:
                if vol_confirm:
                    if is_choppy or (is_trending and not price_above_1w_hma):
                        desired_signal = -SIZE
        
        # === STOPLOSS CHECK (3 ATR) ===
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
        
        # === EXIT CONDITIONS ===
        exit_triggered = False
        
        if in_position and position_side > 0:
            # Long exit: price reaches R3 (take profit zone)
            if close[i] >= r3[i]:
                exit_triggered = True
            # Or RSI overbought in choppy
            if is_choppy and rsi_val > 70:
                exit_triggered = True
        
        if in_position and position_side < 0:
            # Short exit: price reaches S3 (take profit zone)
            if close[i] <= s3[i]:
                exit_triggered = True
            # Or RSI oversold in choppy
            if is_choppy and rsi_val < 30:
                exit_triggered = True
        
        if exit_triggered:
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
                    stop_price = entry_price - 3.0 * entry_atr
                else:
                    stop_price = entry_price + 3.0 * entry_atr
            else:
                pass  # Maintain position
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