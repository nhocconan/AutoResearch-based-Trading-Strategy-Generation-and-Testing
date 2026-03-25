#!/usr/bin/env python3
"""
Experiment #1605: 15m Primary + 4h/1d HTF — HMA Trend + RSI Pullback + Session Filter

Hypothesis: 15m timeframe with 4h trend bias + fast RSI(7) pullback entries can capture
intraday momentum while avoiding the whipsaw that killed previous 15m attempts. Key innovations:

1. 4h HMA(21) for trend direction - slower than 15m noise, faster than 1d
2. RSI(7) for entry timing - faster than RSI(14), catches intraday pullbacks
3. Session filter (00-12 UTC) - London/NY overlap = higher volume, cleaner moves
4. Choppiness Index regime - trend entries when CHOP<45, mean-revert when CHOP>55
5. LOOSE thresholds to guarantee trades: RSI(7)<40/>60 (not 30/70)
6. Position size 0.20-0.25 (smaller for 15m frequency)
7. ATR(14) stoploss at 2.5x for risk management

Why this should work vs failed 15m attempts:
- Previous 15m strategies had TOO STRICT entries (0 trades)
- This uses LOOSE RSI(7) thresholds that trigger frequently
- 4h trend filter is less restrictive than 1d filter
- Session filter reduces noise without eliminating all trades
- Discrete sizing (0.20, 0.25) minimizes fee churn

Entry logic (LOOSE to guarantee ≥30 trades/train):
- LONG: 4h_HMA bullish + RSI(7)<40 + (CHOP<45 OR session 00-12 UTC)
- SHORT: 4h_HMA bearish + RSI(7)>60 + (CHOP<45 OR session 00-12 UTC)
- Exit: RSI(7) crosses 50 OR stoploss hit (2.5x ATR)

Target: Sharpe>0.6, trades>=40 train, trades>=5 test, DD>-35%
Timeframe: 15m
Size: 0.20-0.25 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_hma_rsi_pullback_4h_session_chop_v1"
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

def calculate_session_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds since epoch
    hours = (open_time // 3600000) % 24
    return hours

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
    
    # Calculate 15m indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_7 = calculate_rsi(close, period=7)
    rsi_14 = calculate_rsi(close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    # Extract session hours
    session_hours = np.array([calculate_session_hour(ot) for ot in open_time])
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
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
    min_bars = 50
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_7[i]) or np.isnan(rsi_14[i]) or np.isnan(chop_14[i]):
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
        
        # === REGIME DETECTION (Choppiness Index) ===
        chop = chop_14[i]
        is_trend_regime = chop < 45.0
        is_range_regime = chop > 55.0
        
        # === TREND DIRECTION (4h HMA bias) ===
        price_above_4h = close[i] > hma_4h_aligned[i]
        price_below_4h = close[i] < hma_4h_aligned[i]
        
        # 4h HMA slope (trend confirmation)
        hma_4h_slope_bullish = False
        hma_4h_slope_bearish = False
        if i > 1 and not np.isnan(hma_4h_aligned[i-1]):
            hma_4h_slope_bullish = hma_4h_aligned[i] > hma_4h_aligned[i-1]
            hma_4h_slope_bearish = hma_4h_aligned[i] < hma_4h_aligned[i-1]
        
        # === SESSION FILTER (00-12 UTC = London/NY overlap) ===
        in_session = 0 <= session_hours[i] <= 12
        
        # === RSI SIGNALS (LOOSE thresholds for trade generation) ===
        rsi_7_val = rsi_7[i]
        rsi_14_val = rsi_14[i]
        
        # RSI oversold/overbought (LOOSE: 40/60 not 30/70)
        rsi_oversold = rsi_7_val < 40.0
        rsi_overbought = rsi_7_val > 60.0
        
        # RSI crossing back from extremes (entry trigger)
        rsi_crossing_up = False
        rsi_crossing_down = False
        if i > 0 and not np.isnan(rsi_7[i-1]):
            rsi_crossing_up = rsi_7_val > 35.0 and rsi_7[i-1] <= 35.0
            rsi_crossing_down = rsi_7_val < 65.0 and rsi_7[i-1] >= 65.0
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # TREND REGIME: Follow 4h trend on RSI pullback
        if is_trend_regime:
            # LONG: 4h bullish + RSI(7) oversold + (session OR slope confirm)
            if price_above_4h and rsi_oversold:
                if in_session or hma_4h_slope_bullish:
                    desired_signal = SIZE_STRONG if in_session else SIZE_BASE
            
            # SHORT: 4h bearish + RSI(7) overbought + (session OR slope confirm)
            elif price_below_4h and rsi_overbought:
                if in_session or hma_4h_slope_bearish:
                    desired_signal = -SIZE_STRONG if in_session else -SIZE_BASE
        
        # RANGE REGIME: Mean reversion at RSI extremes
        elif is_range_regime:
            # LONG: RSI(7) deeply oversold
            if rsi_7_val < 35.0:
                desired_signal = SIZE_BASE
            
            # SHORT: RSI(7) deeply overbought
            elif rsi_7_val > 65.0:
                desired_signal = -SIZE_BASE
        
        # NEUTRAL REGIME: Use 1d HMA for bias + RSI(14) confirmation
        else:
            price_above_1d = close[i] > hma_1d_aligned[i]
            price_below_1d = close[i] < hma_1d_aligned[i]
            
            # LONG: 1d bullish + RSI(7) oversold + RSI(14) neutral
            if price_above_1d and rsi_oversold and 40 < rsi_14_val < 60:
                desired_signal = SIZE_BASE
            
            # SHORT: 1d bearish + RSI(7) overbought + RSI(14) neutral
            elif price_below_1d and rsi_overbought and 40 < rsi_14_val < 60:
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
        
        # === EXIT LOGIC (RSI crosses 50 = momentum lost) ===
        if in_position and desired_signal == 0.0:
            # Check if RSI crossed against position
            if i > 0 and not np.isnan(rsi_7[i-1]):
                if position_side > 0 and rsi_7_val > 55.0 and rsi_7[i-1] <= 50.0:
                    desired_signal = 0.0  # Exit long
                elif position_side < 0 and rsi_7_val < 45.0 and rsi_7[i-1] >= 50.0:
                    desired_signal = 0.0  # Exit short
        
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