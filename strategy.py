#!/usr/bin/env python3
"""
Experiment #1461: 15m Primary + 1h/4h/1d HTF — Camarilla Pivot Mean-Reversion

Hypothesis: 15m timeframe with Camarilla pivot levels provides high-probability
mean-reversion entries when combined with HTF trend filters. Camarilla R3/S3
levels act as natural support/resistance for intraday reversals.

Key components:
1. 1d HMA(21) for major trend bias (only trade with daily trend)
2. 4h HMA(16) for intermediate momentum confirmation
3. 15m Camarilla pivot levels (R3/S3 mean-reversion, R4/S4 breakout)
4. RSI(7) for entry timing (oversold/overbought extremes)
5. ATR(14) trailing stoploss at 2.5x
6. Session filter: prefer 00-12 UTC (London+NY overlap)
7. Discrete sizing: 0.15, 0.20, 0.25 (minimize fee churn)

Why this should work:
- Camarilla levels are proven for crypto intraday mean-reversion
- HTF filters prevent counter-trend trades (major killer on 15m)
- RSI(7) provides timely entries at pivot levels
- Session filter avoids low-liquidity Asian session whipsaws
- 15m TF with HTF bias = ~50-80 trades/year (fee-efficient)

Entry logic (LOOSE enough for trades, strict enough for quality):
- LONG: 1d_HMA bullish + 4h_HMA bullish + price near S3 + RSI(7)<25
- SHORT: 1d_HMA bearish + 4h_HMA bearish + price near R3 + RSI(7)>75
- BREAKOUT LONG: 1d bullish + price breaks R4 + volume confirmation
- BREAKOUT SHORT: 1d bearish + price breaks S4 + volume confirmation

Target: Sharpe>0.6, trades>=40 train, trades>=5 test, DD>-35%
Timeframe: 15m
Size: 0.15-0.25 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_camarilla_pivot_rsi_hma_1h4h1d_session_v1"
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

def calculate_camarilla_pivots(high, low, close, prev_close):
    """
    Camarilla Pivot Levels
    R4 = C + (H-L) * 1.5000  (breakout level)
    R3 = C + (H-L) * 1.2500  (mean-reversion short)
    R2 = C + (H-L) * 1.1666
    R1 = C + (H-L) * 1.0833
    S1 = C - (H-L) * 1.0833
    S2 = C - (H-L) * 1.1666
    S3 = C - (H-L) * 1.2500  (mean-reversion long)
    S4 = C - (H-L) * 1.5000  (breakout level)
    """
    n = len(close)
    r4 = np.full(n, np.nan, dtype=np.float64)
    r3 = np.full(n, np.nan, dtype=np.float64)
    s3 = np.full(n, np.nan, dtype=np.float64)
    s4 = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(1, n):
        prev_h = high[i-1]
        prev_l = low[i-1]
        prev_c = prev_close[i-1] if i > 0 else close[i-1]
        
        range_val = prev_h - prev_l
        
        r4[i] = prev_c + range_val * 1.5000
        r3[i] = prev_c + range_val * 1.2500
        s3[i] = prev_c - range_val * 1.2500
        s4[i] = prev_c - range_val * 1.5000
    
    return r4, r3, s3, s4

def calculate_session_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds since epoch
    hours = (open_time // (1000 * 60 * 60)) % 24
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1h = get_htf_data(prices, '1h')
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=16)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1h_raw = calculate_hma(df_1h['close'].values, period=16)
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h_raw)
    
    # Calculate 15m indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_7 = calculate_rsi(close, period=7)
    rsi_14 = calculate_rsi(close, period=14)
    
    # Camarilla pivots (need previous day's OHLC)
    r4, r3, s3, s4 = calculate_camarilla_pivots(high, low, close, close)
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session hours
    session_hours = calculate_session_hour(open_time)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.15
    SIZE_MED = 0.20
    SIZE_STRONG = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup period
    min_bars = 150
    
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
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(r3[i]) or np.isnan(s3[i]) or np.isnan(r4[i]) or np.isnan(s4[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF TREND BIAS ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        hma_4h_bullish = close[i] > hma_4h_aligned[i]
        hma_4h_bearish = close[i] < hma_4h_aligned[i]
        
        hma_1h_bullish = close[i] > hma_1h_aligned[i]
        hma_1h_bearish = close[i] < hma_1h_aligned[i]
        
        # === SESSION FILTER (prefer 00-12 UTC) ===
        is_prime_session = 0 <= session_hours[i] <= 12
        is_asian_session = 18 <= session_hours[i] or session_hours[i] <= 2
        
        # === CAMARILLA LEVEL PROXIMITY ===
        near_s3 = abs(close[i] - s3[i]) / close[i] < 0.005  # within 0.5%
        near_r3 = abs(close[i] - r3[i]) / close[i] < 0.005  # within 0.5%
        broke_r4 = close[i] > r4[i]
        broke_s4 = close[i] < s4[i]
        
        # === RSI EXTREMES ===
        rsi_oversold = rsi_7[i] < 28
        rsi_overbought = rsi_7[i] > 72
        rsi_extreme_os = rsi_7[i] < 20
        rsi_extreme_ob = rsi_7[i] > 80
        
        # === VOLUME CONFIRMATION ===
        vol_above_avg = volume[i] > vol_ma[i] * 1.2 if not np.isnan(vol_ma[i]) else False
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # MEAN-REVERSION LONG: HTF bullish + near S3 + RSI oversold
        if price_above_1d and hma_4h_bullish:
            if near_s3 and rsi_oversold:
                if is_prime_session:
                    desired_signal = SIZE_MED
                else:
                    desired_signal = SIZE_BASE
        
        # MEAN-REVERSION SHORT: HTF bearish + near R3 + RSI overbought
        elif price_below_1d and hma_4h_bearish:
            if near_r3 and rsi_overbought:
                if is_prime_session:
                    desired_signal = -SIZE_MED
                else:
                    desired_signal = -SIZE_BASE
        
        # BREAKOUT LONG: Strong HTF bullish + breaks R4 + volume
        if price_above_1d and hma_4h_bullish and hma_1h_bullish:
            if broke_r4 and vol_above_avg and rsi_7[i] < 75:
                desired_signal = max(desired_signal, SIZE_STRONG)
        
        # BREAKOUT SHORT: Strong HTF bearish + breaks S4 + volume
        if price_below_1d and hma_4h_bearish and hma_1h_bearish:
            if broke_s4 and vol_above_avg and rsi_7[i] > 25:
                desired_signal = min(desired_signal, -SIZE_STRONG)
        
        # EXTREME REVERSION (override HTF if RSI extreme enough)
        if rsi_extreme_os and near_s3:
            desired_signal = max(desired_signal, SIZE_BASE)
        if rsi_extreme_ob and near_r3:
            desired_signal = min(desired_signal, -SIZE_BASE)
        
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
        elif desired_signal >= SIZE_MED * 0.9:
            final_signal = SIZE_MED
        elif desired_signal <= -SIZE_MED * 0.9:
            final_signal = -SIZE_MED
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