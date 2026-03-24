#!/usr/bin/env python3
"""
Experiment #996: 30m Primary + 4h/1d HTF — CHOP Regime + RSI Entries + Session Filter

Hypothesis: 30m timeframe with strict confluence (HTF trend + CHOP regime + RSI + session)
will generate FEW high-quality trades (40-80/year) avoiding fee drag that killed lower TF strategies.

Key innovations:
1. 4h HMA(21) for intermediate trend direction (call ONCE before loop)
2. 1d HMA(21) for higher timeframe bias (call ONCE before loop)
3. Choppiness Index(14) on 30m: CHOP>61.8=range (mean revert), CHOP<38.2=trend
4. RSI(14) extremes for entry timing (not CRSI - simpler, more reliable)
5. Session filter: ONLY trade 08-20 UTC (high liquidity, avoid Asian chop)
6. ATR(14) 2.5x trailing stop for risk management
7. Size: 0.20 discrete (conservative to survive 2022 crash)

Why this should work:
- 30m captures intraday swings without 15m noise
- Session filter removes 60% of low-quality signals (Asian session chop)
- CHOP regime avoids trend strategies in 2022 bottom whipsaw
- HTF bias prevents counter-trend trades
- Few trades = less fee drag (critical for 30m timeframe)

Entry conditions (strict confluence for few trades):
- LONG = 4h bull + 1d bull + CHOP>61.8 + RSI<30 + session 08-20 UTC
- SHORT = 4h bear + 1d bear + CHOP>61.8 + RSI>70 + session 08-20 UTC
- TREND continuation: CHOP<38.2 + RSI pullback to 40-60 + HTF aligned

Target: Sharpe>0.45, trades>=30 train, trades>=5 test, DD>-40%, trades<100/year
Timeframe: 30m
Size: 0.20 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_chop_rsi_session_4h1d_v1"
timeframe = "30m"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=np.float64)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1].astype(np.float64)
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility and stops"""
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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = range/choppy (mean revert)
    CHOP < 38.2 = trending (trend follow)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr_sum += max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
        
        if highest_high > lowest_low and tr_sum > 1e-10:
            chop[i] = 100.0 * np.log10((highest_high - lowest_low) / tr_sum) / np.log10(period)
    
    return chop

def is_session_active(open_time_unix_ms):
    """
    Session filter: only trade 08-20 UTC (high liquidity)
    open_time_unix_ms: Binance open_time in milliseconds
    Returns True if within active session
    """
    # Convert ms to hours UTC
    hour_utc = (open_time_unix_ms // (1000 * 60 * 60)) % 24
    return 8 <= hour_utc < 20

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE = 0.20  # Conservative size for 30m (few trades, low fee drag)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Trade counter for debugging
    trade_count = 0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(chop_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (08-20 UTC only) ===
        session_active = is_session_active(open_time[i])
        
        # === HTF BIAS (4h + 1d HMA) ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # Strong bias: both 4h and 1d aligned
        htf_bull = htf_4h_bull and htf_1d_bull
        htf_bear = htf_4h_bear and htf_1d_bear
        
        # === REGIME DETECTION (CHOP) ===
        is_ranging = chop_14[i] > 61.8  # Mean reversion regime
        is_trending = chop_14[i] < 38.2  # Trend following regime
        
        # === RSI EXTREMES ===
        rsi_oversold = rsi_14[i] < 30
        rsi_overbought = rsi_14[i] > 70
        rsi_neutral_long = 40 <= rsi_14[i] <= 55  # Pullback in uptrend
        rsi_neutral_short = 45 <= rsi_14[i] <= 60  # Pullback in downtrend
        
        # === ENTRY LOGIC (STRICT CONFLUENCE) ===
        desired_signal = 0.0
        
        # LONG: range regime + HTF bull + RSI oversold + session active
        if htf_bull and is_ranging and rsi_oversold and session_active:
            desired_signal = SIZE
            trade_count += 1
        
        # LONG: trend regime + HTF bull + RSI pullback + session active
        elif htf_bull and is_trending and rsi_neutral_long and session_active:
            desired_signal = SIZE
            trade_count += 1
        
        # SHORT: range regime + HTF bear + RSI overbought + session active
        elif htf_bear and is_ranging and rsi_overbought and session_active:
            desired_signal = -SIZE
            trade_count += 1
        
        # SHORT: trend regime + HTF bear + RSI pullback + session active
        elif htf_bear and is_trending and rsi_neutral_short and session_active:
            desired_signal = -SIZE
            trade_count += 1
        
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
        if desired_signal >= SIZE * 0.9:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.9:
            final_signal = -SIZE
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