#!/usr/bin/env python3
"""
Experiment #1587: 6h Primary + 1d HTF — ADX Regime + RSI Entry + Volume Filter

Hypothesis: ADX-based regime detection provides cleaner trend/range separation than
Choppiness Index for 6h timeframe. Combined with loose RSI thresholds (35/65 vs 30/70)
and 1d HMA bias, this should generate consistent trades while avoiding whipsaws.

Key innovations vs previous 6h attempts:
1. ADX REGIME SWITCH: ADX>25=trend (follow 1d HMA), ADX<20=range (RSI mean-revert)
   Hysteresis prevents rapid regime flipping (enter 25, exit 18)
2. LOOSE RSI THRESHOLDS: 35/65 instead of 30/70 to guarantee ≥30 trades/year
3. VOLUME FILTER: Only required for trend breakouts, NOT for mean-reversion entries
   (mean-reversion often happens on low volume capitulation)
4. 1d HMA BIAS: Prevents major counter-trend positions in strong trends
5. ATR STOPLOSS: 2.5x ATR trailing stop via signal→0

Why this should beat mtf_6h_triple_hma_kama_roc_1w1d_v1 (Sharpe=0.575):
- ADX regime clearer than HMA crossover for trend detection
- Looser RSI = more trades = better statistics without fee drag
- Volume filter only on breakouts reduces false signals
- 6h TF = good balance between 4h (too many trades) and 12h (too few)

Entry logic (LOOSE to guarantee trades):
- LONG trend: ADX>25 + 1d_HMA bullish + RSI>35 + close>EMA21
- SHORT trend: ADX>25 + 1d_HMA bearish + RSI<65 + close<EMA21
- LONG range: ADX<20 + RSI<35 + price<BB_lower (no volume filter)
- SHORT range: ADX<20 + RSI>65 + price>BB_upper (no volume filter)

Target: Sharpe>0.6, trades>=40 train, trades>=5 test, DD>-35%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_adx_regime_rsi_1d_v1"
timeframe = "6h"
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

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index - measures trend strength
    ADX > 25 = trending, ADX < 20 = ranging
    """
    n = len(close)
    if n < period * 3:
        return np.full(n, np.nan)
    
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    tr = np.zeros(n, dtype=np.float64)
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(0, high[i] - high[i-1])
        else:
            plus_dm[i] = 0
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(0, low[i-1] - low[i])
        else:
            minus_dm[i] = 0
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = np.zeros(n, dtype=np.float64)
    minus_di = np.zeros(n, dtype=np.float64)
    
    mask = atr > 0
    plus_di[mask] = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[mask] / atr[mask]
    minus_di[mask] = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[mask] / atr[mask]
    
    dx = np.zeros(n, dtype=np.float64)
    di_sum = plus_di + minus_di
    mask2 = di_sum > 0
    dx[mask2] = 100 * np.abs(plus_di[mask2] - minus_di[mask2]) / di_sum[mask2]
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    return adx

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

def calculate_ema(close, period):
    """Exponential Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    ema = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    return ema

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

def calculate_volume_ratio(volume, period=20):
    """Current volume vs average volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / vol_avg
    
    return vol_ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    adx_14 = calculate_adx(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    ema_21 = calculate_ema(close, period=21)
    bb_upper, bb_mid, bb_lower = calculate_bollinger(close, period=20, std_mult=2.0)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    
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
    
    # ADX regime hysteresis tracking
    prev_adx_regime = 0  # 0=neutral, 1=trend, 2=range
    
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
        
        if np.isnan(adx_14[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(bb_lower[i]) or np.isnan(ema_21[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (ADX with hysteresis) ===
        adx = adx_14[i]
        
        # Hysteresis: enter trend at 25, exit at 18; enter range at 20, exit at 27
        if prev_adx_regime == 1:  # was trend
            if adx < 18:
                current_regime = 2  # switch to range
            else:
                current_regime = 1  # stay trend
        elif prev_adx_regime == 2:  # was range
            if adx > 27:
                current_regime = 1  # switch to trend
            else:
                current_regime = 2  # stay range
        else:  # neutral/initial
            if adx > 25:
                current_regime = 1
            elif adx < 20:
                current_regime = 2
            else:
                current_regime = 0
        
        prev_adx_regime = current_regime
        is_trend_regime = (current_regime == 1)
        is_range_regime = (current_regime == 2)
        
        # === TREND DIRECTION (1d HMA bias) ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # === RSI SIGNALS (LOOSE thresholds for trade frequency) ===
        rsi = rsi_14[i]
        rsi_oversold = rsi < 35  # Loose: 35 instead of 30
        rsi_overbought = rsi > 65  # Loose: 65 instead of 70
        
        # === PRICE POSITION ===
        price_above_ema = close[i] > ema_21[i]
        price_below_ema = close[i] < ema_21[i]
        
        # === BOLLINGER BAND TOUCH ===
        bb_touch_lower = close[i] <= bb_lower[i] * 1.005
        bb_touch_upper = close[i] >= bb_upper[i] * 0.995
        
        # === VOLUME CONFIRMATION (trend breakouts only) ===
        vol_confirmed = vol_ratio[i] > 1.2 if not np.isnan(vol_ratio[i]) else False
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # TREND REGIME: Follow 1d HMA direction with RSI confirmation
        if is_trend_regime:
            # LONG: 1d bullish + RSI not overbought + price above EMA21
            if price_above_1d and not rsi_overbought and price_above_ema:
                desired_signal = SIZE_STRONG if vol_confirmed else SIZE_BASE
            
            # SHORT: 1d bearish + RSI not oversold + price below EMA21
            elif price_below_1d and not rsi_oversold and price_below_ema:
                desired_signal = -SIZE_STRONG if vol_confirmed else -SIZE_BASE
        
        # RANGE REGIME: Mean reversion at Bollinger extremes
        elif is_range_regime:
            # LONG: RSI oversold + price at BB lower (NO volume filter for mean-rev)
            if rsi_oversold and bb_touch_lower:
                desired_signal = SIZE_BASE
            
            # SHORT: RSI overbought + price at BB upper (NO volume filter for mean-rev)
            elif rsi_overbought and bb_touch_upper:
                desired_signal = -SIZE_BASE
        
        # NEUTRAL REGIME: Wait for clearer signal
        else:
            # Only enter if strong RSI extreme
            if rsi < 30:
                desired_signal = SIZE_BASE
            elif rsi > 70:
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