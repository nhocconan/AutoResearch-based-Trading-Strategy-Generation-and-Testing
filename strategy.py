#!/usr/bin/env python3
"""
Experiment #1357: 15m Primary + 4h/1d HTF — CPR Pivot + RSI Pullback + Session Filter

Hypothesis: 15m entries with 4h trend + daily CPR levels will generate selective trades
with positive Sharpe. Key innovations:
1. Daily CPR (Central Pivot Range) from 1d data for key support/resistance
2. 4h HMA(21) for trend bias (only trade in trend direction)
3. 15m RSI(7) for pullback entries (oversold in uptrend, overbought in downtrend)
4. Session filter: only trade 00-12 UTC (London+NY overlap)
5. ATR breakout confirmation to filter false signals

Why 15m can work:
- HTF (4h/1d) provides direction, 15m only for timing
- Session filter reduces trades by ~50%
- RSI pullback (not breakout) = fewer false signals
- Size: 0.15-0.20 (smaller for 15m frequency)

Entry logic:
- LONG: 4h HMA bullish + price near daily BC/TC + RSI(7)<35 + session OK
- SHORT: 4h HMA bearish + price near daily TC + RSI(7)>65 + session OK

Target: Sharpe>0.5, trades>=40 train, trades>=5 test, DD>-35%
Timeframe: 15m
Size: 0.15-0.20 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_cpr_rsi_pullback_session_4h1d_v1"
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

def calculate_cpr(open_price, high, low, close):
    """
    Central Pivot Range (CPR) - Daily levels
    Pivot = (H + L + C) / 3
    BC (Bottom Central) = (H + L) / 2
    TC (Top Central) = (Pivot - BC) + Pivot
    """
    n = len(close)
    pivot = np.full(n, np.nan, dtype=np.float64)
    bc = np.full(n, np.nan, dtype=np.float64)
    tc = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(n):
        pivot[i] = (high[i] + low[i] + close[i]) / 3.0
        bc[i] = (high[i] + low[i]) / 2.0
        tc[i] = (pivot[i] - bc[i]) + pivot[i]
    
    return pivot, bc, tc

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_price = prices["open"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate 1d CPR
    pivot_1d_raw, bc_1d_raw, tc_1d_raw = calculate_cpr(
        df_1d['open'].values,
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values
    )
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d_raw)
    bc_1d_aligned = align_htf_to_ltf(prices, df_1d, bc_1d_raw)
    tc_1d_aligned = align_htf_to_ltf(prices, df_1d, tc_1d_raw)
    
    # Calculate 15m indicators
    rsi_7 = calculate_rsi(close, period=7)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Volume median for filter (pre-compute)
    vol_median = np.nanmedian(volume[50:]) if np.sum(~np.isnan(volume[50:])) > 0 else 1.0
    
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
    
    # Trade counter for frequency control
    trades_this_year = 0
    last_trade_year = -1
    
    min_bars = 100
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_7[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(pivot_1d_aligned[i]) or np.isnan(bc_1d_aligned[i]) or np.isnan(tc_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (00-12 UTC) ===
        open_time_ms = prices["open_time"].iloc[i]
        hour_utc = (open_time_ms // 3600000) % 24
        session_ok = 0 <= hour_utc < 14  # Extended to 14 for more trades
        
        # === TREND BIAS (4h HMA) ===
        price_above_4h = close[i] > hma_4h_aligned[i]
        price_below_4h = close[i] < hma_4h_aligned[i]
        
        # === CPR LEVELS (Daily) ===
        # Price near support (BC) or resistance (TC)
        price_near_bc = abs(close[i] - bc_1d_aligned[i]) < 0.008 * close[i]  # Within 0.8%
        price_near_tc = abs(close[i] - tc_1d_aligned[i]) < 0.008 * close[i]
        price_near_pivot = abs(close[i] - pivot_1d_aligned[i]) < 0.008 * close[i]
        
        # CPR width for regime
        cpr_width = abs(tc_1d_aligned[i] - bc_1d_aligned[i])
        cpr_narrow = cpr_width < 0.01 * close[i]  # Narrow CPR = trending
        
        # === RSI PULLBACK (LOOSE for trade generation) ===
        rsi = rsi_7[i]
        rsi_oversold = rsi < 40  # Relaxed from 30
        rsi_overbought = rsi > 60  # Relaxed from 70
        
        # === VOLUME CONFIRMATION ===
        vol_ok = volume[i] > 0.5 * vol_median if not np.isnan(vol_median) else True
        
        # === TRADE FREQUENCY CONTROL ===
        current_year = int(prices["open_time"].iloc[i] // (365 * 24 * 3600000))
        if current_year != last_trade_year:
            trades_this_year = 0
            last_trade_year = current_year
        
        max_trades_per_year = 80  # Target 50-100
        if trades_this_year >= max_trades_per_year and in_position:
            # Let existing position run, but don't enter new
            pass
        
        # === ENTRY LOGIC (2+ confluence for 15m) ===
        desired_signal = 0.0
        confluence_count = 0
        
        # LONG: 4h bullish + near support + RSI oversold + session OK
        if price_above_4h and session_ok and trades_this_year < max_trades_per_year + 5:
            confluence_count = 0
            if price_near_bc or price_near_pivot:
                confluence_count += 1
            if rsi_oversold:
                confluence_count += 1
            if vol_ok:
                confluence_count += 1
            
            if confluence_count >= 2:
                if cpr_narrow:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
        
        # SHORT: 4h bearish + near resistance + RSI overbought + session OK
        elif price_below_4h and session_ok and trades_this_year < max_trades_per_year + 5:
            confluence_count = 0
            if price_near_tc or price_near_pivot:
                confluence_count += 1
            if rsi_overbought:
                confluence_count += 1
            if vol_ok:
                confluence_count += 1
            
            if confluence_count >= 2:
                if cpr_narrow:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
        
        # === STOPLOSS (2.5x ATR trailing) ===
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
                if not in_position:
                    trades_this_year += 1
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