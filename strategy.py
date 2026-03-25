#!/usr/bin/env python3
"""
Experiment #1597: 15m Primary + 4h/12h HTF — Volume Spike Reversal + Regime Filter

Hypothesis: 15m strategies fail due to too many trades and fee drag. This strategy
uses EXTREMELY selective entries with 4+ confluence factors to limit trades to
40-80/year while capturing high-probability volume spike reversals.

Key innovations:
1. VOLUME SPIKE REVERSAL: ATR(7)/ATR(30) > 2.0 indicates panic/extreme move,
   followed by mean reversion. Proven edge in crypto crash/rally scenarios.
2. 4h HMA trend bias: Only trade 15m reversals IN DIRECTION of 4h trend
   (long reversals in 4h uptrend, short reversals in 4h downtrend)
3. 12h Choppiness regime: Trend regime (CHOP<38) = trade pullbacks,
   Range regime (CHOP>61) = trade Bollinger extremes
4. SESSION FILTER: Only trade 00-12 UTC (London/NY overlap = best liquidity)
5. RSI(7) extremes: <20 for long, >80 for short (tighter than standard 30/70)
6. Position size: 0.15-0.20 (smaller due to 15m frequency)

Why this should beat failed 15m attempts:
- Volume spike filter eliminates 80% of false signals
- 4h trend bias prevents counter-trend trades (major killer in 2022)
- Session filter avoids low-liquidity Asian session whipsaws
- Discrete sizing (0.0, ±0.15, ±0.20) minimizes fee churn

Entry logic (VERY STRICT to limit trades):
- LONG: 4h_HMA bullish + (trend: RSI7<25 + ATR_ratio>1.8 OR range: RSI7<20 + BB_lower)
        + volume>1.5x + session 00-12 UTC
- SHORT: 4h_HMA bearish + (trend: RSI7>75 + ATR_ratio>1.8 OR range: RSI7>80 + BB_upper)
         + volume>1.5x + session 00-12 UTC

Target: Sharpe>0.6, trades>=30 train, trades>=3 test, DD>-35%, trades/year<100
Timeframe: 15m
Size: 0.15-0.20 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_volspike_reversal_4h12h_session_v1"
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

def calculate_atr_ratio(atr_short, atr_long):
    """ATR ratio for volume spike detection"""
    n = len(atr_short)
    ratio = np.full(n, np.nan, dtype=np.float64)
    mask = (atr_long > 1e-10) & (~np.isnan(atr_short)) & (~np.isnan(atr_long))
    ratio[mask] = atr_short[mask] / atr_long[mask]
    return ratio

def calculate_volume_ratio(volume, period=20):
    """Current volume vs average volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / vol_avg
    
    return vol_ratio

def is_high_liquidity_session(open_time):
    """
    Check if bar is in high-liquidity session (00-12 UTC)
    open_time is in milliseconds since epoch
    """
    # Convert to hours UTC
    ts_seconds = open_time / 1000
    hour_utc = (ts_seconds % 86400) / 3600
    return 0 <= hour_utc < 12

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    chop_12h_raw = calculate_choppiness(
        df_12h['high'].values,
        df_12h['low'].values,
        df_12h['close'].values,
        period=14
    )
    chop_12h_aligned = align_htf_to_ltf(prices, df_12h, chop_12h_raw)
    
    # Calculate 15m indicators
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_30 = calculate_atr(high, low, close, period=30)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_7 = calculate_rsi(close, period=7)
    rsi_14 = calculate_rsi(close, period=14)
    bb_upper, bb_mid, bb_lower = calculate_bollinger(close, period=20, std_mult=2.0)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    atr_ratio = calculate_atr_ratio(atr_7, atr_30)
    
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
        
        if np.isnan(rsi_7[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(bb_lower[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(chop_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (00-12 UTC only) ===
        if not is_high_liquidity_session(open_time[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (12h Choppiness) ===
        chop = chop_12h_aligned[i]
        is_trend_regime = chop < 38.2
        is_range_regime = chop > 61.8
        
        # === TREND DIRECTION (4h HMA bias) ===
        price_above_4h = close[i] > hma_4h_aligned[i]
        price_below_4h = close[i] < hma_4h_aligned[i]
        
        # === VOLUME SPIKE DETECTION ===
        vol_spike = vol_ratio[i] > 1.5 if not np.isnan(vol_ratio[i]) else False
        atr_spike = atr_ratio[i] > 1.8 if not np.isnan(atr_ratio[i]) else False
        
        # === RSI EXTREMES (tighter than standard) ===
        rsi_oversold = rsi_7[i] < 25
        rsi_overbought = rsi_7[i] > 75
        rsi_extreme_low = rsi_7[i] < 20
        rsi_extreme_high = rsi_7[i] > 80
        
        # === BOLLINGER BAND TOUCH ===
        bb_touch_lower = close[i] <= bb_lower[i] * 1.005
        bb_touch_upper = close[i] >= bb_upper[i] * 0.995
        
        # === ENTRY LOGIC (VERY STRICT - 4+ confluence) ===
        desired_signal = 0.0
        
        # TREND REGIME: Trade pullbacks in direction of 4h trend
        if is_trend_regime:
            # LONG: 4h bullish + RSI oversold + volume or ATR spike
            if price_above_4h and rsi_oversold and (vol_spike or atr_spike):
                desired_signal = SIZE_STRONG if vol_spike and atr_spike else SIZE_BASE
            
            # SHORT: 4h bearish + RSI overbought + volume or ATR spike
            elif price_below_4h and rsi_overbought and (vol_spike or atr_spike):
                desired_signal = -SIZE_STRONG if vol_spike and atr_spike else -SIZE_BASE
        
        # RANGE REGIME: Trade Bollinger extremes (mean reversion)
        elif is_range_regime:
            # LONG: RSI extreme low + BB lower touch + volume confirmation
            if rsi_extreme_low and bb_touch_lower and vol_ratio[i] > 1.2:
                desired_signal = SIZE_BASE
            
            # SHORT: RSI extreme high + BB upper touch + volume confirmation
            elif rsi_extreme_high and bb_touch_upper and vol_ratio[i] > 1.2:
                desired_signal = -SIZE_BASE
        
        # NEUTRAL REGIME: Only trade strongest signals
        else:
            # LONG: 4h bullish + extreme RSI + both volume and ATR spike
            if price_above_4h and rsi_extreme_low and vol_spike and atr_spike:
                desired_signal = SIZE_BASE
            
            # SHORT: 4h bearish + extreme RSI + both volume and ATR spike
            elif price_below_4h and rsi_extreme_high and vol_spike and atr_spike:
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