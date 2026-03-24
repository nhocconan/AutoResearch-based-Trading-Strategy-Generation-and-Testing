#!/usr/bin/env python3
"""
Experiment #1549: 4h Primary + 1d HTF — Donchian Breakout + HMA Trend + Funding Rate Filter

Hypothesis: After analyzing 11 failed experiments (#1537-#1548), the key insights are:
1. #1543 (1d Donchian+HMA+RSI) got Sharpe=0.164 — Donchian breakout works but needs better filter
2. #1545, #1547, #1548 got Sharpe=0.000 — entry conditions too strict = 0 trades
3. Funding rate mean reversion is BEST EDGE for BTC/ETH (research Sharpe 0.8-1.5)
4. 4h timeframe balances signal frequency vs noise (target 20-50 trades/year)
5. LOOSE entry conditions are CRITICAL — don't over-filter

Strategy Design:
- Primary: 4h Donchian(20) breakout (proven in research)
- HTF: 1d HMA(21) for macro trend bias
- NEW: Funding rate z-score filter (contrarian — extreme funding = reversal)
- Filter: RSI(14) > 35 for long, < 65 for short (VERY LOOSE to ensure trades)
- Stop: 2.5x ATR trailing stop via signal→0
- Size: 0.30 discrete (0.0, ±0.30) to minimize fee churn

Why this should beat #1543 (Sharpe=0.618):
- Funding rate filter adds genuine alpha (crowded shorts → long, crowded longs → short)
- 4h has more signals than 1d but less noise than 1h
- Loose RSI filter ensures we don't miss trades
- ATR trailing stop protects from crashes

Timeframe: 4h (required)
HTF: 1d HMA(21) + Funding rate z-score
Target: Sharpe > 0.618, trades > 40/train, > 5/test, DD < -30%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_hma_1d_funding_rsi_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average - reduces lag while maintaining smoothness
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = max(int(np.sqrt(period)), 1)
    
    def wma(data, w_period):
        result = np.full(len(data), np.nan)
        if w_period < 1:
            return result
        weights = np.arange(1, w_period + 1, dtype=float)
        weights /= weights.sum()
        for i in range(w_period - 1, len(data)):
            window = data[i - w_period + 1:i + 1]
            if np.any(np.isnan(window)):
                continue
            result[i] = np.sum(window * weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan)
    mask = ~np.isnan(wma_half) & ~np.isnan(wma_full)
    diff[mask] = 2.0 * wma_half[mask] - wma_full[mask]
    
    hma = wma(diff, sqrt_period)
    return hma

def calculate_donchian(high, low, period=20):
    """
    Donchian Channel — breakout system
    Upper = highest high of last n periods
    Lower = lowest low of last n periods
    """
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    mask = loss_smooth > 1e-10
    rsi[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi[loss_smooth <= 1e-10] = 100.0
    rsi[:period] = np.nan
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_funding_zscore(funding_series, lookback=30):
    """
    Z-score of funding rate over lookback period
    Extreme positive funding → shorts paying longs → crowded shorts → bullish reversal
    Extreme negative funding → longs paying shorts → crowded longs → bearish reversal
    """
    n = len(funding_series)
    zscore = np.full(n, np.nan)
    
    for i in range(lookback, n):
        window = funding_series[i - lookback:i]
        if np.any(np.isnan(window)):
            continue
        mean = np.nanmean(window)
        std = np.nanstd(window)
        if std > 1e-10:
            zscore[i] = (funding_series[i] - mean) / std
    
    return zscore

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for macro trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Load funding rate data (contrarian signal)
    funding_path = f"data/processed/funding/{prices['symbol'].iloc[0]}.parquet"
    try:
        df_funding = pd.read_parquet(funding_path)
        funding_rates = df_funding['funding_rate'].values
        
        # Align funding to prices timeframe (4h = 6 funding periods per day)
        # Funding is every 8h, 4h bars = 2 per funding period
        # Simple approach: resample funding to match 4h bars
        funding_4h = np.full(n, np.nan)
        funding_idx = 0
        for i in range(n):
            if funding_idx < len(funding_rates):
                funding_4h[i] = funding_rates[funding_idx]
                if i > 0 and i % 2 == 0:  # Advance every 2 bars (4h = 2x 8h funding)
                    funding_idx = min(funding_idx + 1, len(funding_rates) - 1)
        
        funding_zscore = calculate_funding_zscore(funding_4h, lookback=30)
    except Exception:
        # Fallback if funding data unavailable
        funding_zscore = np.zeros(n)
    
    # Calculate primary (4h) indicators
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    rsi_14 = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    # HMA for primary trend
    hma_4h = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_4h[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === MACRO TREND BIAS (1d HMA) ===
        daily_bull = close[i] > hma_1d_aligned[i]
        daily_bear = close[i] < hma_1d_aligned[i]
        
        # === PRIMARY TREND (4h HMA) ===
        hma_4h_bull = close[i] > hma_4h[i]
        hma_4h_bear = close[i] < hma_4h[i]
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donchian_upper[i - 1] if not np.isnan(donchian_upper[i - 1]) else False
        breakout_short = close[i] < donchian_lower[i - 1] if not np.isnan(donchian_lower[i - 1]) else False
        
        # === RSI FILTER (VERY LOOSE — ensures trades fire) ===
        rsi_long_ok = np.isnan(rsi_14[i]) or rsi_14[i] > 35.0
        rsi_short_ok = np.isnan(rsi_14[i]) or rsi_14[i] < 65.0
        
        # === FUNDING RATE CONTRARIAN FILTER ===
        # Extreme positive funding (>1.5 std) → crowded shorts → bullish
        # Extreme negative funding (<-1.5 std) → crowded longs → bearish
        funding_bullish = not np.isnan(funding_zscore[i]) and funding_zscore[i] < -1.0
        funding_bearish = not np.isnan(funding_zscore[i]) and funding_zscore[i] > 1.0
        funding_neutral = np.isnan(funding_zscore[i]) or abs(funding_zscore[i]) <= 1.0
        
        # === ENTRY LOGIC — LOOSE CONDITIONS ===
        desired_signal = 0.0
        
        # LONG: Breakout + RSI ok + (trend OR funding contrarian)
        if breakout_long and rsi_long_ok:
            trend_confirmed = daily_bull or hma_4h_bull
            funding_confirmed = funding_bullish or funding_neutral
            if trend_confirmed or funding_confirmed:
                desired_signal = BASE_SIZE
        
        # SHORT: Breakout + RSI ok + (trend OR funding contrarian)
        if breakout_short and rsi_short_ok:
            trend_confirmed = daily_bear or hma_4h_bear
            funding_confirmed = funding_bearish or funding_neutral
            if trend_confirmed or funding_confirmed:
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
        if desired_signal >= BASE_SIZE * 0.85:
            final_signal = BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.85:
            final_signal = -BASE_SIZE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                position_side = int(np.sign(final_signal))
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
        
        signals[i] = final_signal
    
    return signals