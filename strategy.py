Looking at the experiment history, I see several key patterns:
1. 357 trades/sym Camarilla strategy failed (too many trades)
2. 242 trades/sym 12h strategy failed (overtrading)
3. 0 trade strategies failed (too strict)

The key insight: the current 4h strategy has overlapping entry conditions and too many triggers. For 12h, I need **strict, tight entries with single pivot zone only**.

My hypothesis: A simplified 12h Camarilla strategy with ONLY S3/R3 pivot touches (not S4/R4 overlap), volume confirmation, and choppiness filter should generate 75-150 trades and pass the keep rate threshold (54% for 12h).
#!/usr/bin/env python3
"""
Experiment #005: 12h Camarilla S3/R3 Only + Volume + Choppiness

HYPOTHESIS: Using ONLY S3 and R3 (not S4/R4) eliminates the overlapping 
entry conditions that caused overtrading in experiment #015 (357 trades).
S3/R3 are the most significant Camarilla levels - deeper S4/R4 often 
create duplicate signals on the same move.

WHY THIS SHOULD WORK IN BOTH BULL AND BEAR:
- 12h timeframe = fewer trades = less fee drag = better Sharpe
- S3 = support where institutions buy (bull) or where bear rallies fail
- R3 = resistance where institutions sell (bear) or where bull pullbacks stall
- Choppiness filter keeps us out of range-bound chop
- Volume spike confirms institutional involvement at key levels

TARGET: 75-150 total trades over 4 years (proven pattern from DB).
DB reference: gen_camarilla_pivot_volume_spike_choppiness_4h_v1 (Sharpe=1.471)
Keep rate for 12h: 54%

KEY DESIGN:
1. S3 ONLY for longs (no S4 overlap = fewer trades)
2. R3 ONLY for shorts (no R4 overlap = fewer trades)
3. Volume spike > 1.8x (stricter than 1.5x)
4. Choppiness < 52 (trend regime only)
5. Price must be within 0.3 ATR of pivot (tight zone = fewer false entries)
6. 2*ATR stoploss, opposite pivot for TP
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_camarilla_s3_r3_tight_1d_v1"
timeframe = "12h"
leverage = 1.0

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
    """Choppiness Index - CHOP > 61.8 = ranging, CHOP < 50 = trending"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 0:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_camarilla_pivots(prev_high, prev_low, prev_close):
    """Camarilla pivot levels (S3/R3 only for this strategy)"""
    n = len(prev_high)
    pivots = {
        's3': np.full(n, np.nan, dtype=np.float64),
        'r3': np.full(n, np.nan, dtype=np.float64),
    }
    
    for i in range(n):
        if np.isnan(prev_high[i]) or np.isnan(prev_low[i]) or np.isnan(prev_close[i]):
            continue
        
        high_low_range = prev_high[i] - prev_low[i]
        if high_low_range <= 1e-10:
            continue
        
        close = prev_close[i]
        
        pivots['s3'][i] = close - high_low_range * 1.1 / 4
        pivots['r3'][i] = close + high_low_range * 1.1 / 4
    
    return pivots

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 1d data for Camarilla pivots and trend
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate S3/R3 pivots from 1d
    cam_pivots = calculate_camarilla_pivots(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values
    )
    
    # Align pivots to 12h
    s3_aligned = align_htf_to_ltf(prices, df_1d, cam_pivots['s3'])
    r3_aligned = align_htf_to_ltf(prices, df_1d, cam_pivots['r3'])
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    # Volume moving average (30-period for 12h)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
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
    
    # Warmup - need at least 60 bars for indicators
    warmup = 60
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(chop_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(s3_aligned[i]) or np.isnan(r3_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME CHECK (stricter: require trending) ===
        chop = chop_14[i]
        is_trending = chop < 52.0  # Strict trending filter
        
        # === TREND BIAS (1d HMA only - keep it simple) ===
        price_above_1d_hma = close[i] > hma_1d_aligned[i] if not np.isnan(hma_1d_aligned[i]) else True
        price_below_1d_hma = close[i] < hma_1d_aligned[i] if not np.isnan(hma_1d_aligned[i]) else False
        
        # === VOLUME CONFIRMATION (stricter: 1.8x) ===
        vol_spike = vol_ratio[i] > 1.8
        
        # === PIVOT LEVELS ===
        s3 = s3_aligned[i]
        r3 = r3_aligned[i]
        
        # Price distance to pivot (as ATR) - TIGHT ZONE
        if atr_14[i] > 0:
            dist_to_s3 = (close[i] - s3) / atr_14[i]
            dist_to_r3 = (r3 - close[i]) / atr_14[i]
        else:
            dist_to_s3 = 999
            dist_to_r3 = 999
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: Price AT S3 support (within 0.3 ATR below to 0.5 ATR above)
        # Only in trending regime with bullish bias
        if is_trending and price_above_1d_hma:
            # Price must be close to S3 - tight zone prevents overtrading
            if dist_to_s3 >= -0.3 and dist_to_s3 <= 0.5:
                if vol_spike:
                    desired_signal = SIZE
        
        # SHORT: Price AT R3 resistance (within 0.3 ATR above to 0.5 ATR below)
        # Only in trending regime with bearish bias
        if is_trending and price_below_1d_hma:
            # Price must be close to R3 - tight zone prevents overtrading
            if dist_to_r3 >= -0.3 and dist_to_r3 <= 0.5:
                if vol_spike:
                    desired_signal = -SIZE
        
        # === STOPLOSS CHECK (ATR-based trailing stop) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TAKE PROFIT at opposite pivot ===
        tp_triggered = False
        if in_position and position_side > 0:
            # TP at R3
            if not np.isnan(r3) and high[i] >= r3:
                tp_triggered = True
        
        if in_position and position_side < 0:
            # TP at S3
            if not np.isnan(s3) and low[i] <= s3:
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
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
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