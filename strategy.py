#!/usr/bin/env python3
"""
Experiment #1249: 15m Primary + 1h/1d HTF — Mean Reversion with HTF Trend Filter

Hypothesis: After 960+ failed experiments, 15m trend-following has consistently failed
(Sharpe=-4.563, 0.000, 0.000 in recent attempts). The problem: too many whipsaws on
lower TF + fee destruction from overtrading.

NEW APPROACH: Mean reversion ON TOP of HTF trend. This is fundamentally different from
all previous 15m attempts which were pure trend-following.

Logic:
1. 1d HMA(21) = primary trend direction (only trade WITH daily trend)
2. 1h RSI(7) = overbought/oversold extremes for entry timing
3. 15m price vs EMA(21) = confirmation we're at local extreme
4. Volume filter = only trade when volume confirms the move
5. Session bias = prefer 00-12 UTC (London/NY overlap)

Entry conditions (LOOSE enough to guarantee trades):
- LONG: 1d_HMA bullish + 1h_RSI(7) < 30 + price > 15m_EMA(21)
- SHORT: 1d_HMA bearish + 1h_RSI(7) > 70 + price < 15m_EMA(21)

Why this should work:
- 15m timeframe with HTF filter = natural 40-100 trades/year
- Mean reversion entries = higher win rate than trend entries on 15m
- Daily trend filter = avoids counter-trend disasters
- RSI(7) extremes = selective but not impossible (unlike RSI<20)
- Smaller size (0.15-0.20) = accounts for higher frequency

Target: Sharpe>0.5, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 15m
Size: 0.15-0.20 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_mean_reversion_hma_rsi_1h1d_v1"
timeframe = "15m"
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
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = np.nan
    return rsi

def calculate_ema(close, period):
    """Exponential Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    ema = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    return ema

def calculate_volume_sma(volume, period=20):
    """Simple Moving Average of Volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    rsi_1h_raw = calculate_rsi(df_1h['close'].values, period=7)
    rsi_1h_aligned = align_htf_to_ltf(prices, df_1h, rsi_1h_raw)
    
    # Calculate 15m indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    ema_21_15m = calculate_ema(close, period=21)
    vol_sma_20 = calculate_volume_sma(volume, period=20)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.15
    SIZE_STRONG = 0.20
    
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
        
        if np.isnan(ema_21_15m[i]):
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
        
        if np.isnan(rsi_1h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND DIRECTION (Daily HMA) ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # === 1h RSI EXTREMES (Mean Reversion Signal) ===
        rsi_1h = rsi_1h_aligned[i]
        rsi_oversold = rsi_1h < 30.0
        rsi_overbought = rsi_1h > 70.0
        
        # === 15m PRICE POSITION (Local Extreme Confirmation) ===
        price_above_ema = close[i] > ema_21_15m[i]
        price_below_ema = close[i] < ema_21_15m[i]
        
        # === VOLUME CONFIRMATION (Optional boost) ===
        vol_above_avg = False
        if not np.isnan(vol_sma_20[i]) and vol_sma_20[i] > 0:
            vol_above_avg = volume[i] > 1.2 * vol_sma_20[i]
        
        # === SESSION FILTER (Prefer 00-12 UTC) ===
        # Extract hour from open_time (milliseconds timestamp)
        hour_utc = (open_time[i] // 3600000) % 24
        prime_session = 0 <= hour_utc < 12
        
        # === ENTRY LOGIC (Mean Reversion WITH Trend) ===
        desired_signal = 0.0
        
        # LONG: Daily bullish + 1h RSI oversold + price above 15m EMA
        # (RSI oversold suggests pullback complete, ready to resume uptrend)
        if price_above_1d and rsi_oversold and price_above_ema:
            if vol_above_avg and prime_session:
                desired_signal = SIZE_STRONG  # All conditions met
            elif vol_above_avg or prime_session:
                desired_signal = SIZE_BASE  # Partial confirmation
            else:
                desired_signal = SIZE_BASE  # Basic setup
        
        # SHORT: Daily bearish + 1h RSI overbought + price below 15m EMA
        # (RSI overbought suggests rally exhausted, ready to resume downtrend)
        elif price_below_1d and rsi_overbought and price_below_ema:
            if vol_above_avg and prime_session:
                desired_signal = -SIZE_STRONG  # All conditions met
            elif vol_above_avg or prime_session:
                desired_signal = -SIZE_BASE  # Partial confirmation
            else:
                desired_signal = -SIZE_BASE  # Basic setup
        
        # === STOPLOSS CHECK (2.0x ATR trailing) ===
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
        
        signals[i] = final_signal
    
    return signals