#!/usr/bin/env python3
"""
Experiment #1305: 15m Primary + 4h/1d HTF — Daily CPR Breakout with HTF Trend Filter

Hypothesis: 15m strategies have failed due to excessive trades and fee drag. This strategy
uses Daily CPR (Central Pivot Range) levels from 1d HTF as key support/resistance, combined
with 4h HMA trend filter and volume confirmation. Key innovations:

1. Daily CPR from 1d HTF: BC/TC levels act as intraday pivot points (proven in tradfi)
2. 4h HMA(21) for major trend bias (only trade CPR breakouts in trend direction)
3. Volume spike filter: taker_buy_volume ratio > 1.5 for breakout confirmation
4. Session filter: 00-12 UTC only (London+NY overlap, highest liquidity)
5. RSI(7) pullback entry within CPR range for mean-reversion trades
6. VERY selective: require 3+ confluence (HTF trend + CPR level + volume + session)

Why this should work on 15m:
- CPR levels are respected intraday (institutional order flow)
- 4h trend filter prevents counter-trend trades (major failure mode)
- Session filter avoids low-liquidity whipsaws (Asia session)
- Volume confirmation filters false breakouts
- Target: 40-80 trades/year (strict entry conditions)

Entry logic:
- LONG: 4h_HMA bullish + price > TC + volume_spike + session + RSI(7) < 70
- SHORT: 4h_HMA bearish + price < BC + volume_spike + session + RSI(7) > 30
- MEAN REVERT: Inside CPR range + RSI(7) extremes + HTF trend alignment

Timeframe: 15m (FIRST 15m EXPERIMENT)
Size: 0.15-0.20 discrete (smaller for higher frequency)
Target: Sharpe>0.5, trades>=30 train, trades>=5 test, DD>-35%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_cpr_breakout_hma_volume_4h1d_session_v1"
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
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if avg_loss[i] != 0:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0
    
    return rsi

def calculate_cpr_levels(df_1d):
    """
    Calculate Daily CPR (Central Pivot Range) levels from 1d data.
    CPR = [BC, Pivot, TC] where:
    - Pivot = (High + Low + Close) / 3
    - BC (Bottom Central) = (High + Low) / 2
    - TC (Top Central) = (Pivot + BC) / 2
    
    Returns arrays aligned to 1d bars.
    """
    n = len(df_1d)
    high = df_1d['high'].values
    low = df_1d['low'].values
    close = df_1d['close'].values
    
    pivot = np.full(n, np.nan, dtype=np.float64)
    bc = np.full(n, np.nan, dtype=np.float64)
    tc = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(1, n):  # Start from 1 (use previous day's levels)
        pivot[i] = (high[i-1] + low[i-1] + close[i-1]) / 3.0
        bc[i] = (high[i-1] + low[i-1]) / 2.0
        tc[i] = (pivot[i] + bc[i]) / 2.0
    
    return pivot, bc, tc

def calculate_volume_ratio(taker_buy_volume, volume):
    """Calculate taker buy volume ratio (buying pressure)"""
    n = len(volume)
    ratio = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        if volume[i] > 0:
            ratio[i] = taker_buy_volume[i] / volume[i]
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_vol = prices["taker_buy_volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate Daily CPR levels and align to 15m
    pivot_1d, bc_1d, tc_1d = calculate_cpr_levels(df_1d)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    bc_aligned = align_htf_to_ltf(prices, df_1d, bc_1d)
    tc_aligned = align_htf_to_ltf(prices, df_1d, tc_1d)
    
    # Calculate 15m indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_7 = calculate_rsi(close, period=7)
    rsi_14 = calculate_rsi(close, period=14)
    vol_ratio = calculate_volume_ratio(taker_buy_vol, volume)
    
    # Calculate 15m HMA for local trend
    hma_15m = calculate_hma(close, period=21)
    
    # Calculate CPR width (narrow CPR = consolidation, wide = trending)
    cpr_width = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        if not np.isnan(tc_aligned[i]) and not np.isnan(bc_aligned[i]) and close[i] > 0:
            cpr_width[i] = (tc_aligned[i] - bc_aligned[i]) / close[i] * 100.0
    
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
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(pivot_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_15m[i]) or np.isnan(cpr_width[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (00-12 UTC only) ===
        # Convert open_time (ms) to hour
        hour_utc = (open_time[i] // (1000 * 60 * 60)) % 24
        in_session = (hour_utc >= 0 and hour_utc < 12)  # London+NY overlap
        
        # === TREND DIRECTION (4h HMA slope) ===
        hma_4h_slope = 0.0
        if i >= 4 and not np.isnan(hma_4h_aligned[i-4]):
            hma_4h_slope = hma_4h_aligned[i] - hma_4h_aligned[i-4]
        
        trend_bullish = hma_4h_slope > 0
        trend_bearish = hma_4h_slope < 0
        
        # === CPR LEVELS ===
        price_above_tc = close[i] > tc_aligned[i]
        price_below_bc = close[i] < bc_aligned[i]
        price_inside_cpr = close[i] >= bc_aligned[i] and close[i] <= tc_aligned[i]
        
        # Narrow CPR = consolidation (breakout more likely)
        narrow_cpr = cpr_width[i] < 0.5  # Less than 0.5% width
        
        # === VOLUME CONFIRMATION ===
        volume_spike = vol_ratio[i] > 1.5 or vol_ratio[i] < 0.5  # Strong buying or selling
        
        # === 15m LOCAL TREND ===
        price_above_15m = close[i] > hma_15m[i]
        price_below_15m = close[i] < hma_15m[i]
        
        # === ENTRY LOGIC (VERY SELECTIVE - 3+ confluence required) ===
        desired_signal = 0.0
        
        # LONG BREAKOUT: 4h bullish + price > TC + volume spike + session + narrow CPR
        if trend_bullish and price_above_tc and in_session:
            confluence_count = 0
            if volume_spike and vol_ratio[i] > 1.5:
                confluence_count += 1
            if narrow_cpr:
                confluence_count += 1
            if price_above_15m:
                confluence_count += 1
            if rsi_7[i] < 75:  # Not overbought
                confluence_count += 1
            
            if confluence_count >= 3:
                if confluence_count >= 4:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
        
        # SHORT BREAKOUT: 4h bearish + price < BC + volume spike + session + narrow CPR
        elif trend_bearish and price_below_bc and in_session:
            confluence_count = 0
            if volume_spike and vol_ratio[i] < 0.5:
                confluence_count += 1
            if narrow_cpr:
                confluence_count += 1
            if price_below_15m:
                confluence_count += 1
            if rsi_7[i] > 25:  # Not oversold
                confluence_count += 1
            
            if confluence_count >= 3:
                if confluence_count >= 4:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
        
        # MEAN REVERSION LONG: Inside CPR + 4h bullish + RSI(7) oversold + session
        elif trend_bullish and price_inside_cpr and in_session:
            if rsi_7[i] < 25 and rsi_14[i] < 35:
                if vol_ratio[i] > 1.2:  # Buying pressure building
                    desired_signal = SIZE_BASE
        
        # MEAN REVERSION SHORT: Inside CPR + 4h bearish + RSI(7) overbought + session
        elif trend_bearish and price_inside_cpr and in_session:
            if rsi_7[i] > 75 and rsi_14[i] > 65:
                if vol_ratio[i] < 0.8:  # Selling pressure building
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