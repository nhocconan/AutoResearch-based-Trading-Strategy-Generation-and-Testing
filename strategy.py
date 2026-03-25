#!/usr/bin/env python3
"""
Experiment #1534: 1d Primary + 1w HTF — Volatility-Adaptive Regime Strategy

Hypothesis: Daily timeframe with weekly trend filter provides optimal balance of
trade frequency (20-50/year) and signal quality. This strategy combines:

1. 1w HMA(21) for major trend bias — avoids counter-trend in strong moves
2. 1d KAMA(14) for adaptive trend following — adapts speed to volatility
3. ATR Ratio (7/30) for volatility regime — spikes signal mean reversion
4. RSI(14) with asymmetric thresholds — different levels for bull/bear
5. Volume confirmation — taker_buy_volume ratio > 0.55 for longs
6. Choppiness Index(14) for regime switch — trend vs mean-reversion mode

Key innovations vs failed strategies:
- LOOSE entry thresholds (RSI 35/65 not 30/70) to guarantee trades
- Volume confirmation prevents false breakouts
- ATR ratio detects vol spikes (panic = reversion opportunity)
- KAMA adapts to market speed (fast in trends, slow in chop)
- Weekly HMA filter prevents major counter-trend disasters

Entry logic (designed for ≥30 trades/train, ≥5/test):
- LONG trend: 1w_HMA bullish + KAMA rising + RSI>45 + vol_confirm + CHOP<50
- SHORT trend: 1w_HMA bearish + KAMA falling + RSI<55 + vol_confirm + CHOP<50
- LONG reversion: ATR_ratio>2.0 (vol spike) + RSI<40 + price<BB_lower
- SHORT reversion: ATR_ratio>2.0 (vol spike) + RSI>60 + price>BB_upper

Position sizing: 0.25 base, 0.30 strong signals (discrete to minimize fees)
Stoploss: 2.5x ATR trailing stop

Target: Sharpe>0.6, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 1d
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_vol_regime_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_kama(close, period=14, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average
    Adapts smoothing based on market efficiency (trend vs noise)
    """
    n = len(close)
    if n < period + slow_period:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan, dtype=np.float64)
    
    # Calculate Efficiency Ratio (ER)
    er = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        price_change = abs(close[i] - close[i - period])
        noise = 0.0
        for j in range(i - period + 1, i + 1):
            noise += abs(close[j] - close[j - 1])
        if noise > 1e-10:
            er[i] = price_change / noise
        else:
            er[i] = 0.0
    
    # Calculate smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA
    kama[period] = close[period]
    
    for i in range(period + 1, n):
        if not np.isnan(er[i]):
            sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
            kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

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

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain = np.insert(gain, 0, 0)
    loss = np.insert(loss, 0, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    mask = avg_loss != 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppy vs trending
    CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    """
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

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    return upper, sma, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_vol = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF indicators
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 1d indicators
    kama_14 = calculate_kama(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_30 = calculate_atr(high, low, close, period=30)
    rsi_14 = calculate_rsi(close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    bb_upper, bb_mid, bb_lower = calculate_bollinger(close, period=20, std_mult=2.0)
    
    # ATR ratio for volatility spike detection
    atr_ratio = np.full(n, np.nan, dtype=np.float64)
    for i in range(30, n):
        if atr_30[i] > 1e-10:
            atr_ratio[i] = atr_7[i] / atr_30[i]
    
    # Volume confirmation (taker buy ratio)
    vol_ratio = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        if volume[i] > 1e-10:
            vol_ratio[i] = taker_buy_vol[i] / volume[i]
        else:
            vol_ratio[i] = 0.5
    
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
    
    # Warmup period
    min_bars = 100
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(kama_14[i]) or np.isnan(chop_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(bb_lower[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(atr_ratio[i]) or np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION ===
        chop = chop_14[i]
        is_trend_regime = chop < 50.0  # Looser threshold for more trades
        is_vol_spike = atr_ratio[i] > 1.8  # Vol spike = reversion likely
        
        # === TREND DIRECTION (1w HMA bias) ===
        price_above_1w = close[i] > hma_1w_aligned[i]
        price_below_1w = close[i] < hma_1w_aligned[i]
        
        # === KAMA DIRECTION (adaptive trend) ===
        kama_rising = kama_14[i] > kama_14[i-1] if not np.isnan(kama_14[i-1]) else False
        kama_falling = kama_14[i] < kama_14[i-1] if not np.isnan(kama_14[i-1]) else False
        
        # === RSI ===
        rsi = rsi_14[i]
        
        # === VOLUME CONFIRMATION ===
        vol_bullish = vol_ratio[i] > 0.52  # More buyers than sellers
        vol_bearish = vol_ratio[i] < 0.48  # More sellers than buyers
        
        # === BOLLINGER BAND TOUCH ===
        bb_touch_lower = close[i] <= bb_lower[i] * 1.005  # within 0.5% of lower band
        bb_touch_upper = close[i] >= bb_upper[i] * 0.995  # within 0.5% of upper band
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # TREND REGIME: KAMA direction + 1w bias + RSI confirmation
        if is_trend_regime and not is_vol_spike:
            # LONG: 1w bullish + KAMA rising + RSI not overbought + vol confirm
            if price_above_1w and kama_rising and rsi > 40 and rsi < 70 and vol_bullish:
                desired_signal = SIZE_STRONG
            
            # SHORT: 1w bearish + KAMA falling + RSI not oversold + vol confirm
            elif price_below_1w and kama_falling and rsi < 60 and rsi > 30 and vol_bearish:
                desired_signal = -SIZE_STRONG
        
        # VOL SPIKE REGIME: Mean reversion (panic = opportunity)
        elif is_vol_spike:
            # LONG: RSI oversold + price at BB lower (panic sell = buy)
            if rsi < 45 and bb_touch_lower:
                desired_signal = SIZE_BASE
            
            # SHORT: RSI overbought + price at BB upper (panic buy = sell)
            elif rsi > 55 and bb_touch_upper:
                desired_signal = -SIZE_BASE
        
        # NEUTRAL/CHOPPY: Reduced size, tighter conditions
        else:
            # LONG: 1w bullish + RSI moderate + vol confirm
            if price_above_1w and rsi > 45 and rsi < 65 and vol_bullish:
                desired_signal = SIZE_BASE * 0.8
            
            # SHORT: 1w bearish + RSI moderate + vol confirm
            elif price_below_1w and rsi < 55 and rsi > 35 and vol_bearish:
                desired_signal = -SIZE_BASE * 0.8
        
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
        elif desired_signal >= SIZE_BASE * 0.5:
            final_signal = SIZE_BASE * 0.8
        elif desired_signal <= -SIZE_BASE * 0.5:
            final_signal = -SIZE_BASE * 0.8
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