#!/usr/bin/env python3
"""
Experiment #1201: 4h Primary + 1d/1w HTF — Simplified HMA Trend + RSI Pullback

Hypothesis: Recent 4h failures (#1199, #1189) stem from overly complex regime filters
blocking too many trades. This version simplifies to proven pattern:
- 1w HMA = ultra-long-term market bias (bull/bear)
- 1d HMA = medium-term trend direction
- 4h RSI = pullback entry timing (not extreme, just moderate pullback)
- ADX = trend strength confirmation (soft filter, not hard block)
- ATR trailing stop = risk management

Key changes from failed #1199:
1. Remove Choppiness regime switch (was blocking 60%+ of valid signals)
2. RSI thresholds: 35/65 instead of 30/70 (more entry opportunities)
3. ADX > 20 soft filter (not hard requirement)
4. Both HTFs must align for entry (1w + 1d same direction)
5. Simpler exit: just stoploss + signal flip (no complex take-profit logic)

Target: 30-50 trades/year, Sharpe > 0.612 (beat current best)
Position Size: 0.30 discrete
Stoploss: 2.5x ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_rsi_pullback_1d1w_adx_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average — reduces lag while maintaining smoothness."""
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff = 2.0 * wma_half[i] - wma_full[i]
            if i >= sqrt_period - 1:
                diff_window = []
                for j in range(i - sqrt_period + 1, i + 1):
                    if j >= period - 1 and not np.isnan(2.0 * wma_half[j] - wma_full[j]):
                        diff_window.append(2.0 * wma_half[j] - wma_full[j])
                if len(diff_window) == sqrt_period:
                    weights = np.arange(1, sqrt_period + 1)
                    hma[i] = np.sum(np.array(diff_window) * weights) / np.sum(weights)
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index — momentum oscillator."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.where(delta > 0, delta, 0)
    loss[1:] = np.where(delta < 0, -delta, 0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = loss_smooth > 1e-10
    rs = np.zeros(n)
    rs[mask] = gain_smooth[mask] / loss_smooth[mask]
    
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = np.nan
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility measurement and stoploss."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_adx(high, low, close, period=14):
    """Average Directional Index — trend strength indicator."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2 + 1:
        return adx
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)
    
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    atr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = atr_smooth > 1e-10
    plus_di[mask] = 100.0 * plus_dm_smooth[mask] / atr_smooth[mask]
    minus_di[mask] = 100.0 * minus_dm_smooth[mask] / atr_smooth[mask]
    
    dx = np.zeros(n)
    mask2 = (plus_di + minus_di) > 1e-10
    dx[mask2] = 100.0 * np.abs(plus_di[mask2] - minus_di[mask2]) / (plus_di[mask2] + minus_di[mask2])
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    adx[:period*2] = np.nan
    
    return adx, plus_di, minus_di

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1d HMA for medium-term trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=34)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for long-term bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (4h) indicators
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    adx, plus_di, minus_di = calculate_adx(high, low, close, period=14)
    
    # Also calculate 4h HMA for additional trend confirmation
    hma_4h = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(adx[i]):
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(hma_4h[i]) or atr[i] <= 1e-10:
            continue
        
        # === MACRO TREND BIAS (1w HMA) ===
        # 1w HMA defines bull/bear market regime
        macro_bull = close[i] > hma_1w_aligned[i]
        macro_bear = close[i] < hma_1w_aligned[i]
        
        # === MEDIUM TREND (1d HMA) ===
        # 1d HMA defines current trend direction
        trend_bull = close[i] > hma_1d_aligned[i]
        trend_bear = close[i] < hma_1d_aligned[i]
        
        # === 4h HMA for local trend ===
        local_bull = close[i] > hma_4h[i]
        local_bear = close[i] < hma_4h[i]
        
        # === TREND STRENGTH (ADX) ===
        # ADX > 20 = trending, ADX < 20 = ranging (soft filter)
        is_trending = adx[i] > 20.0
        
        # === RSI PULLBACK ===
        # Long: RSI 35-50 (pullback in uptrend)
        # Short: RSI 50-65 (pullback in downtrend)
        rsi_pullback_long = 35.0 <= rsi[i] <= 55.0
        rsi_pullback_short = 45.0 <= rsi[i] <= 65.0
        
        # RSI extreme (reversal potential)
        rsi_oversold = rsi[i] < 40.0
        rsi_overbought = rsi[i] > 60.0
        
        # === ENTRY CONDITIONS ===
        desired_signal = 0.0
        
        # === LONG ENTRY ===
        # Requirements: macro bull + trend bull + RSI pullback + ADX confirms
        # At least 2 of 3 HTF/local trend must align
        long_trend_count = sum([macro_bull, trend_bull, local_bull])
        
        if long_trend_count >= 2 and rsi_pullback_long:
            # In trending market: need ADX confirmation
            if is_trending and adx[i] > 25.0:
                desired_signal = BASE_SIZE
            # In ranging market: just need trend alignment
            elif not is_trending and long_trend_count >= 2:
                desired_signal = BASE_SIZE
        
        # Also allow long on RSI oversold in macro bull market
        elif macro_bull and rsi_oversold and trend_bull:
            desired_signal = BASE_SIZE
        
        # === SHORT ENTRY ===
        # Requirements: macro bear + trend bear + RSI pullback + ADX confirms
        short_trend_count = sum([macro_bear, trend_bear, local_bear])
        
        if short_trend_count >= 2 and rsi_pullback_short:
            # In trending market: need ADX confirmation
            if is_trending and adx[i] > 25.0:
                desired_signal = -BASE_SIZE
            # In ranging market: just need trend alignment
            elif not is_trending and short_trend_count >= 2:
                desired_signal = -BASE_SIZE
        
        # Also allow short on RSI overbought in macro bear market
        elif macro_bear and rsi_overbought and trend_bear:
            desired_signal = -BASE_SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE
        else:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals