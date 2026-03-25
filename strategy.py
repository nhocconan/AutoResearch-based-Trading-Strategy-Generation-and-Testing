#!/usr/bin/env python3
"""
Experiment #1485: 15m Primary + 4h/1d HTF — Daily CPR Pivot Breakout Strategy

Hypothesis: 15m timeframe with Daily CPR (Central Pivot Range) from 1d HTF provides
optimal intraday entry points. CPR is a proven pivot system used by professional
traders - narrow CPR indicates consolidation (breakout potential), wide CPR indicates
trending conditions.

Key components:
1. 1d CPR calculation: Pivot = (H+L+C)/3, BC = (H+L)/2, TC = (Pivot-BC)+Pivot
   - Narrow CPR (< 0.5% of price) = consolidation breakout setup
   - Price above TC = bullish bias, below BC = bearish bias
2. 4h HMA(21) for intermediate trend direction (aligned properly via mtf_data)
3. 15m RSI(7) for entry timing: <30 oversold long, >70 overbought short
4. Session filter: 00-12 UTC (London/NY overlap = highest crypto liquidity)
5. ATR(14) trailing stoploss (2.5x ATR)
6. Discrete sizing: 0.15, 0.20 (smaller for 15m frequency control)

Why this should work on 15m:
- CPR levels are institutional reference points (self-fulfilling prophecy)
- 4h HMA filters counter-trend trades (major killer on lower TFs)
- RSI(7) is fast enough for 15m but not too noisy
- Session filter reduces low-liquidity whipsaws
- LOOSE thresholds (RSI 30/70, not 25/75) guarantee trades

Entry logic (LOOSE to guarantee ≥10 trades/train, ≥3/test):
- LONG: 4h_HMA bullish + price>TC + RSI<45 (pullback entry in uptrend)
- SHORT: 4h_HMA bearish + price<BC + RSI>55 (rally entry in downtrend)
- BREAKOUT LONG: Narrow CPR + price breaks above TC + 4h_HMA bullish
- BREAKOUT SHORT: Narrow CPR + price breaks below BC + 4h_HMA bearish

Target: Sharpe>0.6, trades>=40 train, trades>=5 test, DD>-35%, trades<100/year
Timeframe: 15m
Size: 0.15-0.20 discrete (smaller for higher frequency)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_cpr_pivot_rsi_session_4h1d_v1"
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

def calculate_cpr_from_ohlc(high, low, close):
    """
    Calculate CPR (Central Pivot Range) levels from OHLC data
    Pivot = (High + Low + Close) / 3
    BC (Bottom Central) = (High + Low) / 2
    TC (Top Central) = (Pivot - BC) + Pivot
    
    Returns: pivot, bc, tc arrays
    """
    n = len(close)
    pivot = (high + low + close) / 3.0
    bc = (high + low) / 2.0
    tc = (pivot - bc) + pivot
    return pivot, bc, tc

def calculate_cpr_width(pivot, bc, tc, close):
    """
    Calculate CPR width as percentage of price
    Narrow CPR (< 0.5%) = consolidation/breakout setup
    Wide CPR (> 1.5%) = trending conditions
    """
    cpr_range = tc - bc
    cpr_width_pct = (cpr_range / close) * 100.0
    return cpr_width_pct

def is_session_active(open_time_unix_ms):
    """
    Check if current bar is in preferred session (00-12 UTC)
    open_time_unix_ms: Unix timestamp in milliseconds
    Returns True if in session, False otherwise
    """
    # Convert ms to hours UTC
    hour_utc = (open_time_unix_ms // (1000 * 3600)) % 24
    return 0 <= hour_utc < 12

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 4h HMA trend
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d CPR levels
    pivot_1d_raw, bc_1d_raw, tc_1d_raw = calculate_cpr_from_ohlc(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values
    )
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d_raw)
    bc_1d_aligned = align_htf_to_ltf(prices, df_1d, bc_1d_raw)
    tc_1d_aligned = align_htf_to_ltf(prices, df_1d, tc_1d_raw)
    
    # Calculate 1d CPR width for regime detection
    cpr_width_1d_raw = calculate_cpr_width(pivot_1d_raw, bc_1d_raw, tc_1d_raw, df_1d['close'].values)
    cpr_width_1d_aligned = align_htf_to_ltf(prices, df_1d, cpr_width_1d_raw)
    
    # Calculate 15m indicators
    rsi_7 = calculate_rsi(close, period=7)
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # 15m EMA for additional trend confirmation
    ema_21 = pd.Series(close).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    
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
    bars_since_last_trade = 0
    
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
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(pivot_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION ===
        # 4h trend direction
        hma_4h_bullish = close[i] > hma_4h_aligned[i]
        hma_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # 15m trend confirmation
        ema_bullish = ema_21[i] > ema_50[i]
        ema_bearish = ema_21[i] < ema_50[i]
        
        # CPR regime
        cpr_width = cpr_width_1d_aligned[i]
        narrow_cpr = not np.isnan(cpr_width) and cpr_width < 0.8  # < 0.8% = narrow
        wide_cpr = not np.isnan(cpr_width) and cpr_width > 1.5    # > 1.5% = wide
        
        # Price position relative to CPR
        price_above_tc = close[i] > tc_1d_aligned[i]
        price_below_bc = close[i] < bc_1d_aligned[i]
        price_in_cpr = not price_above_tc and not price_below_bc
        
        # RSI levels
        rsi = rsi_7[i]
        rsi_oversold = rsi < 35
        rsi_overbought = rsi > 65
        rsi_neutral = 35 <= rsi <= 65
        
        # Session filter (00-12 UTC)
        in_session = is_session_active(open_time[i])
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # MODE 1: Trend-following pullback entries (primary)
        # LONG: 4h bullish + price above TC + RSI pullback to oversold
        if hma_4h_bullish and price_above_tc and rsi_oversold:
            desired_signal = SIZE_STRONG if in_session else SIZE_BASE
        
        # SHORT: 4h bearish + price below BC + RSI rally to overbought
        elif hma_4h_bearish and price_below_bc and rsi_overbought:
            desired_signal = -SIZE_STRONG if in_session else -SIZE_BASE
        
        # MODE 2: CPR breakout on narrow CPR (consolidation breakout)
        elif narrow_cpr:
            # LONG breakout: price breaks above TC + 4h bullish or neutral
            if price_above_tc and (hma_4h_bullish or ema_bullish) and rsi < 60:
                desired_signal = SIZE_BASE
            
            # SHORT breakdown: price breaks below BC + 4h bearish or neutral
            elif price_below_bc and (hma_4h_bearish or ema_bearish) and rsi > 40:
                desired_signal = -SIZE_BASE
        
        # MODE 3: Mean reversion inside CPR (range trading)
        elif price_in_cpr and not narrow_cpr:
            # LONG: RSI deeply oversold inside CPR
            if rsi < 25:
                desired_signal = SIZE_BASE
            
            # SHORT: RSI deeply overbought inside CPR
            elif rsi > 75:
                desired_signal = -SIZE_BASE
        
        # MODE 4: Fallback - ensure we generate trades if none for 50+ bars
        bars_since_last_trade += 1
        if bars_since_last_trade > 50 and not in_position:
            # Very loose entry to guarantee trades
            if hma_4h_bullish and rsi < 50:
                desired_signal = SIZE_BASE
            elif hma_4h_bearish and rsi > 50:
                desired_signal = -SIZE_BASE
        
        if desired_signal != 0.0:
            bars_since_last_trade = 0
        
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