#!/usr/bin/env python3
"""
Experiment #1495: 6h Primary + 12h/1d HTF — Keltner-ADX Regime Adaptive

Hypothesis: 6h timeframe offers optimal balance between 4h (too many trades) and 
12h (too few trades). This strategy uses ADX regime detection to switch between
trend-following (Keltner breakout) and mean-reversion (Keltner bounce) based on
market conditions, with 1d HMA for major trend bias.

Key components:
1. 1d HMA(21) for major trend bias (avoid counter-trend in strong trends)
2. 12h ADX(14) for regime detection:
   - ADX > 25 = trending (use Keltner breakout strategy)
   - ADX < 20 = ranging (use Keltner mean-reversion)
   - Between = neutral (reduce position size)
3. 6h Keltner Channel(20, 2.0*ATR) for entry signals
4. 6h RSI(14) for momentum confirmation
5. 6h HMA(16/48) for trend momentum confirmation
6. ATR(14) trailing stoploss (2.5x ATR)
7. Discrete sizing: 0.0, ±0.25, ±0.30 (minimize fee churn)

Why this should work:
- ADX regime adaptation prevents trend strategies from dying in chop
- Keltner Channels are more adaptive than Bollinger (uses ATR not std)
- 6h TF = natural 30-60 trades/year (fee-efficient)
- LOOSE entry thresholds guarantee trades (RSI 35/65, not 30/70)
- 1d HMA filter prevents major counter-trend disasters
- Different from failed CRSI/weekly pivot strategies

Entry logic (LOOSE to guarantee ≥30 trades/train, ≥3/test):
- LONG trend: 1d_HMA bullish + ADX>25 + price>Keltner_upper + HMA16>HMA48 + RSI>45
- SHORT trend: 1d_HMA bearish + ADX>25 + price<Keltner_lower + HMA16<HMA48 + RSI<55
- LONG range: ADX<20 + price<Keltner_lower + RSI<40
- SHORT range: ADX<20 + price>Keltner_upper + RSI>60

Target: Sharpe>0.6, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_keltner_adx_regime_hma_1d12h_v1"
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

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index - measures trend strength
    ADX > 25 = trending, ADX < 20 = ranging
    """
    n = len(close)
    if n < period * 2 + 1:
        return np.full(n, np.nan)
    
    # Calculate True Range and Directional Movement
    tr = np.zeros(n, dtype=np.float64)
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        elif down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    # Smooth using Wilder's method (EMA with span=period)
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di_raw = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_di_raw = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate DI
    plus_di = np.full(n, np.nan, dtype=np.float64)
    minus_di = np.full(n, np.nan, dtype=np.float64)
    
    mask = atr > 1e-10
    plus_di[mask] = 100 * plus_di_raw[mask] / atr[mask]
    minus_di[mask] = 100 * minus_di_raw[mask] / atr[mask]
    
    # Calculate DX and ADX
    dx = np.full(n, np.nan, dtype=np.float64)
    di_sum = plus_di + minus_di
    di_diff = np.abs(plus_di - minus_di)
    mask2 = di_sum > 1e-10
    dx[mask2] = 100 * di_diff[mask2] / di_sum[mask2]
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_keltner(high, low, close, period=20, atr_mult=2.0):
    """Keltner Channels - EMA middle with ATR bands"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    ema = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    atr = calculate_atr(high, low, close, period)
    
    upper = ema + atr_mult * atr
    lower = ema - atr_mult * atr
    
    return upper, ema, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    adx_12h_raw = calculate_adx(df_12h['high'].values, df_12h['low'].values, df_12h['close'].values, period=14)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 6h indicators
    hma_16 = calculate_hma(close, period=16)
    hma_48 = calculate_hma(close, period=48)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    keltner_upper, keltner_mid, keltner_lower = calculate_keltner(high, low, close, period=20, atr_mult=2.0)
    
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
        
        if np.isnan(rsi_14[i]) or np.isnan(hma_16[i]) or np.isnan(hma_48[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(keltner_upper[i]) or np.isnan(keltner_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(adx_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (12h ADX) ===
        adx = adx_12h_aligned[i]
        is_trend_regime = adx > 25.0
        is_range_regime = adx < 20.0
        is_neutral_regime = not is_trend_regime and not is_range_regime
        
        # === TREND DIRECTION (1d HMA bias) ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # === 6h HMA CROSSOVER (trend momentum) ===
        hma_bullish = hma_16[i] > hma_48[i]
        hma_bearish = hma_16[i] < hma_48[i]
        
        # === RSI ===
        rsi = rsi_14[i]
        
        # === KELTNER CHANNEL TOUCH/BREAK ===
        keltner_break_upper = close[i] > keltner_upper[i-1] if not np.isnan(keltner_upper[i-1]) else False
        keltner_break_lower = close[i] < keltner_lower[i-1] if not np.isnan(keltner_lower[i-1]) else False
        keltner_touch_upper = close[i] >= keltner_upper[i] * 0.998
        keltner_touch_lower = close[i] <= keltner_lower[i] * 1.002
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # TREND REGIME: Keltner breakout + HMA confirmation + 1d bias
        if is_trend_regime:
            # LONG: 1d bullish + HMA bullish + Keltner breakout + RSI confirmation
            if price_above_1d and hma_bullish and keltner_break_upper and rsi > 45:
                desired_signal = SIZE_STRONG
            
            # SHORT: 1d bearish + HMA bearish + Keltner breakdown + RSI confirmation
            elif price_below_1d and hma_bearish and keltner_break_lower and rsi < 55:
                desired_signal = -SIZE_STRONG
        
        # RANGE REGIME: Keltner mean reversion + RSI extremes
        elif is_range_regime:
            # LONG: price at Keltner lower + RSI oversold
            if keltner_touch_lower and rsi < 40:
                desired_signal = SIZE_BASE
            
            # SHORT: price at Keltner upper + RSI overbought
            elif keltner_touch_upper and rsi > 60:
                desired_signal = -SIZE_BASE
        
        # NEUTRAL REGIME: Only take strong signals with full confluence
        elif is_neutral_regime:
            # LONG: 1d bullish + HMA bullish + RSI not overbought + Keltner break
            if price_above_1d and hma_bullish and rsi < 65 and keltner_break_upper:
                desired_signal = SIZE_BASE
            
            # SHORT: 1d bearish + HMA bearish + RSI not oversold + Keltner break
            elif price_below_1d and hma_bearish and rsi > 35 and keltner_break_lower:
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