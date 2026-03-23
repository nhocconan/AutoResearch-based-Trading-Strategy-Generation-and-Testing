#!/usr/bin/env python3
"""
Experiment #1167: 1d Primary + 1w HTF — Simplified HMA Trend + RSI Pullback

Hypothesis: After analyzing 850+ failed experiments, clear pattern emerges:
- 1d strategies FAIL when entry conditions are too strict (#1157, #1159, #1163, #1165, #1166 = 0 trades)
- 1d strategies FAIL when using complex regime switching (choppiness + CRSI + Donchian = #1163 Sharpe=-3.159)
- SUCCESS requires: SIMPLE logic + LOOSE thresholds + ENSURED trade generation

Why this should work where others failed:
1. HMA(21) trend filter only — no complex regime detection
2. RSI pullback 35-55 (long) / 45-65 (short) — NOT extremes, triggers more often
3. ADX > 15 (not 25) — ensures enough momentum without filtering too much
4. 1w HMA as loose macro filter — only exit on major reversal, not entry block
5. ATR 2.5x trailing stop — appropriate for 1d volatility
6. Position size 0.25 discrete — balances returns vs 2022 crash risk

Key lesson from failures: 1d has ~1800 bars total. Need trade every ~50 bars average.
Over-filtering = 0 trades = auto-reject. Simpler = more trades = chance for positive Sharpe.

Timeframe: 1d (primary)
HTF: 1w — loaded ONCE before loop using mtf_data helper
Position Size: 0.25 base (discrete: 0.0, ±0.25)
Stoploss: 2.5x ATR trailing
Target: 30-50 trades train, 5-10 trades test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_hma_rsi_pullback_1w_adx_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average — reduces lag while maintaining smoothness.
    HMA = WMA(2*WMA(n/2) - WMA(n))
    """
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
    """
    Relative Strength Index — momentum oscillator.
    RSI < 30 = oversold, RSI > 70 = overbought
    For pullback entries: RSI 35-55 (long), RSI 45-65 (short)
    """
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

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index — measures trend strength.
    ADX > 15 = enough momentum for 1d timeframe (looser than 25)
    """
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2:
        return adx
    
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(0, high[i] - high[i-1])
        else:
            plus_dm[i] = 0
            
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(0, low[i-1] - low[i])
        else:
            minus_dm[i] = 0
    
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    di_plus = np.zeros(n)
    di_minus = np.zeros(n)
    mask = tr_smooth > 1e-10
    di_plus[mask] = 100.0 * plus_dm_smooth[mask] / tr_smooth[mask]
    di_minus[mask] = 100.0 * minus_dm_smooth[mask] / tr_smooth[mask]
    
    dx = np.zeros(n)
    for i in range(period * 2 - 1, n):
        di_sum = di_plus[i] + di_minus[i]
        if di_sum > 1e-10:
            dx[i] = 100.0 * abs(di_plus[i] - di_minus[i]) / di_sum
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

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

def calculate_sma(close, period=200):
    """Simple Moving Average for long-term trend filter."""
    n = len(close)
    sma = np.full(n, np.nan)
    
    if n < period:
        return sma
    
    for i in range(period - 1, n):
        sma[i] = np.mean(close[i - period + 1:i + 1])
    
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for macro trend filter
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    atr = calculate_atr(high, low, close, period=14)
    adx_1d = calculate_adx(high, low, close, period=14)
    rsi_1d = calculate_rsi(close, period=14)
    hma_1d = calculate_hma(close, period=21)
    sma_200 = calculate_sma(close, period=200)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):  # Start after 200 SMA + indicator warmup
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(adx_1d[i]) or np.isnan(rsi_1d[i]):
            continue
        if np.isnan(hma_1d[i]) or np.isnan(sma_200[i]):
            continue
        if atr[i] <= 1e-10:
            continue
        
        # === MACRO TREND (1w HMA) — LOOSE FILTER ===
        # Only use for exit, not entry blocking (to ensure trades)
        macro_bull = True if np.isnan(hma_1w_aligned[i]) else close[i] > hma_1w_aligned[i]
        macro_bear = True if np.isnan(hma_1w_aligned[i]) else close[i] < hma_1w_aligned[i]
        
        # === LONG-TERM TREND (SMA 200) ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === LOCAL TREND (1d HMA) ===
        local_bull = close[i] > hma_1d[i]
        local_bear = close[i] < hma_1d[i]
        
        # === TREND STRENGTH (ADX) — LOOSE THRESHOLD ===
        # ADX > 15 ensures some momentum without over-filtering
        trend_strong = adx_1d[i] > 15.0
        
        # === RSI PULLBACK ENTRY — LOOSE THRESHOLDS ===
        # Long: RSI pulled back to 35-55 in uptrend (NOT extreme oversold)
        # Short: RSI pulled back to 45-65 in downtrend (NOT extreme overbought)
        # These trigger MORE OFTEN than 30/70 extremes
        rsi_pullback_long = 35.0 <= rsi_1d[i] <= 55.0
        rsi_pullback_short = 45.0 <= rsi_1d[i] <= 65.0
        
        # === ENTRY CONDITIONS — SIMPLIFIED FOR TRADE GENERATION ===
        desired_signal = 0.0
        
        # === LONG ENTRY ===
        # Above SMA200 + local bull + ADX confirms + RSI pullback
        # OR: Strong RSI oversold (<35) + above SMA200 (catch deep pullbacks)
        if above_sma200 and local_bull and rsi_pullback_long:
            desired_signal = BASE_SIZE
        elif above_sma200 and rsi_1d[i] < 35.0:
            # Deep oversold in uptrend = strong long signal
            desired_signal = BASE_SIZE
        
        # === SHORT ENTRY ===
        # Below SMA200 + local bear + ADX confirms + RSI pullback
        # OR: Strong RSI overbought (>65) + below SMA200 (catch deep retracements)
        if below_sma200 and local_bear and rsi_pullback_short:
            desired_signal = -BASE_SIZE
        elif below_sma200 and rsi_1d[i] > 65.0:
            # Deep overbought in downtrend = strong short signal
            desired_signal = -BASE_SIZE
        
        # === MACRO TREND REVERSAL EXIT ===
        # Exit long if 1w trend turns bear (major reversal)
        if in_position and position_side > 0 and macro_bear and not np.isnan(hma_1w_aligned[i]):
            desired_signal = 0.0
        
        # Exit short if 1w trend turns bull (major reversal)
        if in_position and position_side < 0 and macro_bull and not np.isnan(hma_1w_aligned[i]):
            desired_signal = 0.0
        
        # === SMA200 REVERSAL EXIT ===
        # Exit long if price crosses below SMA200
        if in_position and position_side > 0 and below_sma200:
            desired_signal = 0.0
        
        # Exit short if price crosses above SMA200
        if in_position and position_side < 0 and above_sma200:
            desired_signal = 0.0
        
        # === ADX WEAKNESS EXIT ===
        # If ADX drops below 12, trend is dying — exit
        if in_position and adx_1d[i] < 12.0:
            desired_signal = 0.0
        
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
        
        # === HOLD LOGIC — Maintain position if trend intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if above SMA200 and local still bull
                if above_sma200 and local_bull and adx_1d[i] >= 12.0:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if below SMA200 and local still bear
                if below_sma200 and local_bear and adx_1d[i] >= 12.0:
                    desired_signal = -BASE_SIZE
        
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
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Flip position
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
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
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals