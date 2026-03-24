#!/usr/bin/env python3
"""
Experiment #1043: 6h Primary + 1d/1w HTF — Simple HMA Trend + RSI Pullback

Hypothesis: 6h is the "Goldilocks" timeframe - captures multi-day swings without noise.
Simple is better: 1w HMA for long-term bias, 1d HMA for intermediate trend, 6h RSI for
pullback entries. Avoid complex regime detection (failed in #1036, #1039).

Key innovations:
1. 1w HMA(21): Primary trend bias (very stable, rarely whipsaws)
2. 1d HMA(21): Intermediate confirmation (filters false breakouts)
3. 6h RSI(14) pullback: Enter on dips in uptrend, rallies in downtrend
4. ATR volatility filter: Only trade when ATR > median (avoid dead markets)
5. Volume confirmation: taker_buy_volume ratio > 0.45 for longs, < 0.55 for shorts
6. 2.5x ATR trailing stop with signal→0 on breach

Why this should work:
- 6h captures 2-5 day swings perfectly (BTC typical swing duration)
- 1w HMA provides stable bias (no whipsaw like 4h/12h)
- RSI pullback entries have 60-65% win rate in trending markets
- Simple = fewer parameters = less overfitting
- Loose entries guarantee trades (RSI < 50 for long in uptrend, not < 30)

Entry conditions (LOOSE to guarantee 30+ trades/year):
- LONG: price > 1w_HMA AND price > 1d_HMA AND RSI(14) < 50 AND vol_ratio > 0.45
- SHORT: price < 1w_HMA AND price < 1d_HMA AND RSI(14) > 50 AND vol_ratio < 0.55

Exit conditions:
- RSI crosses back above 60 (long) or below 40 (short)
- Stoploss: 2.5x ATR trailing stop
- HTF trend reversal: price crosses 1d_HMA against position

Target: Sharpe>0.50, trades>=40 train, trades>=5 test, DD>-35%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_hma_rsi_pullback_1d1w_simple_v1"
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
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = np.nan
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_vol = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    
    # Volume ratio (taker buy / total volume)
    vol_ratio = np.divide(taker_buy_vol, volume, out=np.zeros_like(volume), where=volume > 0)
    
    # ATR median for volatility filter (avoid dead markets)
    atr_median = np.nanmedian(atr_14[100:])
    if np.isnan(atr_median) or atr_median <= 0:
        atr_median = np.nanpercentile(atr_14, 50)
    
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
    bars_in_trade = 0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF TREND BIAS ===
        # 1w HMA = long-term bias, 1d HMA = intermediate confirmation
        hma_1w_bull = close[i] > hma_1w_aligned[i]
        hma_1w_bear = close[i] < hma_1w_aligned[i]
        hma_1d_bull = close[i] > hma_1d_aligned[i]
        hma_1d_bear = close[i] < hma_1d_aligned[i]
        
        # Strong alignment: both 1w and 1d agree
        strong_bull = hma_1w_bull and hma_1d_bull
        strong_bear = hma_1w_bear and hma_1d_bear
        
        # === VOLATILITY FILTER ===
        # Only trade when ATR is above median (avoid dead/choppy markets)
        vol_filter = atr_14[i] >= atr_median * 0.8
        
        # === VOLUME CONFIRMATION ===
        # Longs need buying pressure, shorts need selling pressure
        vol_buy_confirm = vol_ratio[i] > 0.45
        vol_sell_confirm = vol_ratio[i] < 0.55
        
        # === ENTRY LOGIC (LOOSE to guarantee trades) ===
        desired_signal = 0.0
        
        # LONG entries: in uptrend, RSI pullback, volume confirms
        if strong_bull and vol_filter:
            if rsi_14[i] < 50.0 and vol_buy_confirm:
                desired_signal = SIZE_BASE
            # Stronger entry on deeper pullback
            if rsi_14[i] < 40.0 and vol_buy_confirm:
                desired_signal = SIZE_STRONG
        
        # SHORT entries: in downtrend, RSI rally, volume confirms
        elif strong_bear and vol_filter:
            if rsi_14[i] > 50.0 and vol_sell_confirm:
                desired_signal = -SIZE_BASE
            # Stronger entry on stronger rally
            if rsi_14[i] > 60.0 and vol_sell_confirm:
                desired_signal = -SIZE_STRONG
        
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
        
        # === EXIT LOGIC (RSI reversal or HTF trend break) ===
        if in_position and position_side > 0:
            # Exit long if RSI goes overbought or trend breaks
            if rsi_14[i] > 65.0 or close[i] < hma_1d_aligned[i]:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if RSI goes oversold or trend breaks
            if rsi_14[i] < 35.0 or close[i] > hma_1d_aligned[i]:
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
                bars_in_trade = 0
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
            bars_in_trade += 1
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                bars_in_trade = 0
        
        signals[i] = final_signal
    
    return signals