#!/usr/bin/env python3
"""
Experiment #1035: 6h Primary + 12h/1d HTF — Fisher Transform + BB Regime + Volume Confirm

Hypothesis: 6h timeframe captures multi-day swings without noise. Using Ehlers Fisher Transform
for reversal detection (proven in bear/range markets) combined with Bollinger Band Width regime
filter and volume confirmation will outperform RSI/Choppiness-based strategies.

Key innovations:
1. Ehlers Fisher Transform (period=9): Normalizes price to -1 to +1, catches reversals early
   Long when Fisher crosses above -1.5, Short when crosses below +1.5
2. BB Width Percentile Regime: BBW in bottom 20% = squeeze (breakout mode), 
   BBW in top 80% = expansion (mean revert mode)
3. Volume Confirmation: taker_buy_volume / volume > 0.55 for long, < 0.45 for short
4. HTF Trend Bias: 1d HMA(21) + 12h HMA(21) alignment for direction filter
5. Asymmetric Entry: Only long when price > 1d_HMA, only short when price < 1d_HMA
6. ATR(14) 2.5x trailing stop for risk management

Why this should work on 6h:
- Fisher Transform excels in bear/range markets (2022 crash, 2025 test period)
- BB Width regime avoids breakout failures in low-vol compression
- Volume confirmation filters false signals (critical for 6h trade quality)
- 6h captures 3-5 day swings, targeting 30-60 trades/year
- HTF bias (12h/1d) prevents counter-trend trades that destroy Sharpe

Entry conditions (LOOSE to guarantee trades):
- LONG: Fisher>-1.5 + cross above -1.0 + price>1d_HMA + vol_ratio>0.52 + BBW not extreme
- SHORT: Fisher<1.5 + cross below 1.0 + price<1d_HMA + vol_ratio<0.48 + BBW not extreme

Target: Sharpe>0.45, trades>=30 train, trades>=5 test, DD>-40%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_bbregime_volume_12h1d_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while smoothing"""
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

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution
    Makes turning points clearly visible by constraining output to -1 to +1
    
    Formula:
    1. Calculate typical price: (high + low) / 2
    2. Normalize: 0.66 * ((price - LL) / (HH - LL) - 0.5) + 0.66 * prev
    3. Fisher: 0.5 * ln((1 + normalized) / (1 - normalized))
    4. Signal line: 1-period lag of Fisher
    """
    n = len(close)
    if n < period + 5:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.full(n, np.nan, dtype=np.float64)
    fisher_signal = np.full(n, np.nan, dtype=np.float64)
    
    # Calculate typical price
    typical = (high + low) / 2.0
    
    # Normalize price (Ehlers method)
    normalized = np.full(n, np.nan, dtype=np.float64)
    normalized[0] = 0.0
    
    for i in range(1, n):
        # Find highest high and lowest low over period
        if i >= period:
            hh = np.max(high[i-period+1:i+1])
            ll = np.min(low[i-period+1:i+1])
        else:
            hh = np.max(high[:i+1])
            ll = np.min(low[:i+1])
        
        price_range = hh - ll
        if price_range > 1e-10:
            norm_val = 0.66 * ((typical[i] - ll) / price_range - 0.5) + 0.67 * normalized[i-1]
            # Clamp to prevent division issues
            norm_val = np.clip(norm_val, -0.99, 0.99)
            normalized[i] = norm_val
        else:
            normalized[i] = normalized[i-1] if i > 0 else 0.0
    
    # Calculate Fisher Transform
    for i in range(1, n):
        if abs(normalized[i]) < 0.99:
            fisher[i] = 0.5 * np.log((1.0 + normalized[i]) / (1.0 - normalized[i]))
            # Clamp fisher value
            fisher[i] = np.clip(fisher[i], -2.5, 2.5)
        else:
            fisher[i] = fisher[i-1] if i > 0 else 0.0
    
    # Signal line is 1-period lag
    fisher_signal[1:] = fisher[:-1]
    fisher_signal[0] = fisher[0]
    
    return fisher, fisher_signal

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Bollinger Bands with width calculation"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    # Rolling mean and std
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    width = (upper - lower) / sma  # BB Width as percentage
    
    return upper, lower, width

def calculate_bbw_percentile(bb_width, lookback=100):
    """Calculate percentile rank of BB Width over lookback period"""
    n = len(bb_width)
    bbw_pct = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(lookback, n):
        window = bb_width[i-lookback+1:i+1]
        if not np.any(np.isnan(window)):
            count_below = np.sum(window[:-1] < bb_width[i])
            bbw_pct[i] = 100.0 * count_below / (lookback - 1)
    
    return bbw_pct

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_vol = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, period=9)
    bb_upper, bb_lower, bb_width = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    bbw_pct = calculate_bbw_percentile(bb_width, lookback=100)
    
    # Volume ratio (taker buy / total volume)
    vol_ratio = np.divide(taker_buy_vol, volume, out=np.zeros_like(volume), where=volume > 0)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(bb_width[i]) or np.isnan(bbw_pct[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (BB Width Percentile) ===
        is_squeeze = bbw_pct[i] < 25.0  # Low vol compression - expect breakout
        is_expansion = bbw_pct[i] > 75.0  # High vol - expect mean reversion
        is_normal = 25.0 <= bbw_pct[i] <= 75.0  # Normal regime
        
        # === HTF TREND BIAS ===
        price_above_1d_hma = close[i] > hma_1d_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_aligned[i]
        price_above_12h_hma = close[i] > hma_12h_aligned[i]
        price_below_12h_hma = close[i] < hma_12h_aligned[i]
        
        # Strong trend alignment
        strong_bull = price_above_1d_hma and price_above_12h_hma and hma_1d_aligned[i] > hma_12h_aligned[i]
        strong_bear = price_below_1d_hma and price_below_12h_hma and hma_1d_aligned[i] < hma_12h_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_bullish = vol_ratio[i] > 0.52
        vol_bearish = vol_ratio[i] < 0.48
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_cross_up = fisher[i] > -1.0 and fisher_signal[i] <= -1.0
        fisher_cross_down = fisher[i] < 1.0 and fisher_signal[i] >= 1.0
        fisher_extreme_low = fisher[i] < -1.5
        fisher_extreme_high = fisher[i] > 1.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG entries (only when price above 1d HMA for trend alignment)
        if price_above_1d_hma:
            # Squeeze breakout long
            if is_squeeze and fisher_cross_up and vol_bullish:
                desired_signal = SIZE_STRONG
            # Normal regime Fisher reversal long
            elif is_normal and fisher_extreme_low and fisher[i] > fisher_signal[i] and vol_bullish:
                desired_signal = SIZE_BASE
            # Strong trend continuation
            elif strong_bull and fisher[i] > -0.5 and fisher[i] < 1.0 and vol_bullish:
                desired_signal = SIZE_BASE
        
        # SHORT entries (only when price below 1d HMA for trend alignment)
        if price_below_1d_hma:
            # Squeeze breakdown short
            if is_squeeze and fisher_cross_down and vol_bearish:
                desired_signal = -SIZE_STRONG
            # Normal regime Fisher reversal short
            elif is_normal and fisher_extreme_high and fisher[i] < fisher_signal[i] and vol_bearish:
                desired_signal = -SIZE_BASE
            # Strong trend continuation
            elif strong_bear and fisher[i] < 0.5 and fisher[i] > -1.0 and vol_bearish:
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
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
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
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
        
        signals[i] = final_signal
    
    return signals