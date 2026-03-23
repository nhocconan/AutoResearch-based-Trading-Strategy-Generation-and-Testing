#!/usr/bin/env python3
"""
Experiment #1021: 4h Primary + 1d HTF — ADX Trend + RSI Pullback + ATR Stop

Hypothesis: After 740+ failed experiments, the pattern is clear:
1. Complex regime switching (chop/fisher/vol) → 0 trades or negative Sharpe
2. Simple trend + pullback logic works best on 4h timeframe
3. BTC/ETH need asymmetric bias (easier long, harder short for bear markets)
4. MUST relax entry thresholds to guarantee >=30 trades on train

This strategy uses PROVEN components:
1. 1d HMA21: Macro trend bias (price above = bullish bias, below = bearish)
2. ADX(14) + DI: Trend strength confirmation (ADX>20 = trending, not chop)
3. RSI(14) pullback: Entry timing (RSI<45 in uptrend, RSI>55 in downtrend)
4. ATR(14) trailing stop: 2.5x for risk management

Key differences from failed experiments:
- SIMPLER logic: 3 conditions max (not 5+ regime filters)
- RELAXED RSI thresholds: 45/55 not 30/70 (more trades)
- RELAXED ADX: >20 not >25 (ADX>25 is too rare)
- Asymmetric bias: long when price>1d_HMA, short when price<1d_HMA*0.98
- Discrete signals: 0.0, ±0.25, ±0.30 to minimize fee churn

Why this should work:
- 4h timeframe targets 30-60 trades/year (sweet spot for fee vs signal)
- ADX filters out choppy whipsaw (the #1 killer of trend strategies)
- RSI pullback entries catch retracements in trends (higher win rate)
- 1d HMA provides stable trend bias (less noise than 4h HMA)

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 4h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_adx_rsi_1d_hma_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_adx_di(high, low, close, period=14):
    """
    ADX + DI+ and DI- calculation.
    ADX > 20 = trending market
    ADX < 20 = ranging/choppy market
    """
    n = len(close)
    adx = np.full(n, np.nan)
    dip = np.full(n, np.nan)
    dim = np.full(n, np.nan)
    
    if n < period * 2 + 1:
        return adx, dip, dim
    
    tr = np.zeros(n)
    dm_plus = np.zeros(n)
    dm_minus = np.zeros(n)
    
    # Calculate True Range and Directional Movement
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        if high[i] - high[i-1] > low[i-1] - low[i]:
            dm_plus[i] = max(high[i] - high[i-1], 0)
        else:
            dm_plus[i] = 0
            
        if low[i-1] - low[i] > high[i] - high[i-1]:
            dm_minus[i] = max(low[i-1] - low[i], 0)
        else:
            dm_minus[i] = 0
    
    # Smooth TR, DM+, DM- using Wilder's method (EMA-like)
    tr_smooth = np.zeros(n)
    dp_smooth = np.zeros(n)
    dm_smooth = np.zeros(n)
    
    # Initial sum for first period
    tr_smooth[period] = np.sum(tr[1:period+1])
    dp_smooth[period] = np.sum(dm_plus[1:period+1])
    dm_smooth[period] = np.sum(dm_minus[1:period+1])
    
    # Wilder's smoothing
    for i in range(period + 1, n):
        tr_smooth[i] = tr_smooth[i-1] - tr_smooth[i-1]/period + tr[i]
        dp_smooth[i] = dp_smooth[i-1] - dp_smooth[i-1]/period + dm_plus[i]
        dm_smooth[i] = dm_smooth[i-1] - dm_smooth[i-1]/period + dm_minus[i]
    
    # Calculate DI+ and DI-
    for i in range(period, n):
        if tr_smooth[i] > 1e-10:
            dip[i] = 100 * dp_smooth[i] / tr_smooth[i]
            dim[i] = 100 * dm_smooth[i] / tr_smooth[i]
    
    # Calculate ADX
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = dip[i] + dim[i]
        if di_sum > 1e-10:
            dx[i] = 100 * abs(dip[i] - dim[i]) / di_sum
    
    # Smooth DX to get ADX
    adx[period*2] = np.sum(dx[period:period*2+1]) / period
    for i in range(period*2 + 1, n):
        adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
    
    return adx, dip, dim

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.where(delta > 0, delta, 0)
    loss[1:] = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    for i in range(period, n):
        if avg_loss[i] == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / (avg_loss[i] + 1e-10)
            rsi[i] = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
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

def calculate_hma(series, period):
    """Hull Moving Average."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA21 for macro trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    adx_4h, dip_4h, dim_4h = calculate_adx_di(high, low, close, period=14)
    rsi_4h = calculate_rsi(close, period=14)
    atr_4h = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(adx_4h[i]) or np.isnan(rsi_4h[i]) or np.isnan(atr_4h[i]):
            continue
        if np.isnan(hma_1d_aligned[i]) or atr_4h[i] <= 1e-10:
            continue
        
        # === MACRO TREND (1d HMA21) ===
        # Asymmetric: long when price > 1d_HMA, short when price < 1d_HMA * 0.98
        trend_bull = close[i] > hma_1d_aligned[i]
        trend_bear = close[i] < hma_1d_aligned[i] * 0.98
        
        # === TREND STRENGTH (ADX) ===
        trend_strong = adx_4h[i] > 20  # Trending (not chop)
        
        # === RSI PULLBACK ENTRY ===
        # Long: RSI pulled back to 35-45 in uptrend
        # Short: RSI rallied to 55-65 in downtrend
        rsi_oversold = rsi_4h[i] < 45
        rsi_overbought = rsi_4h[i] > 55
        rsi_cross_long = rsi_4h[i] > 40 and rsi_4h[i-1] <= 40
        rsi_cross_short = rsi_4h[i] < 60 and rsi_4h[i-1] >= 60
        
        # === DI CONFIRMATION ===
        di_bull = dip_4h[i] > dim_4h[i] if not np.isnan(dip_4h[i]) and not np.isnan(dim_4h[i]) else False
        di_bear = dim_4h[i] > dip_4h[i] if not np.isnan(dip_4h[i]) and not np.isnan(dim_4h[i]) else False
        
        desired_signal = 0.0
        
        # === LONG ENTRIES ===
        if trend_bull and trend_strong:
            # Primary long: trend + ADX + RSI pullback
            if rsi_oversold and di_bull:
                desired_signal = BASE_SIZE
            elif rsi_cross_long and di_bull:
                desired_signal = REDUCED_SIZE
        
        # === SHORT ENTRIES ===
        if trend_bear and trend_strong:
            # Primary short: trend + ADX + RSI rally
            if rsi_overbought and di_bear:
                desired_signal = -BASE_SIZE
            elif rsi_cross_short and di_bear:
                desired_signal = -REDUCED_SIZE
        
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
        
        # === EXIT CONDITIONS ===
        # Exit long if trend reverses or RSI extremely overbought
        if in_position and position_side > 0:
            if not trend_bull or rsi_4h[i] > 70:
                desired_signal = 0.0
        
        # Exit short if trend reverses or RSI extremely oversold
        if in_position and position_side < 0:
            if not trend_bear or rsi_4h[i] < 30:
                desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if conditions intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if trend still bull and RSI not extreme
                if trend_bull and rsi_4h[i] < 65:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if trend still bear and RSI not extreme
                if trend_bear and rsi_4h[i] > 35:
                    desired_signal = -BASE_SIZE
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE if desired_signal >= BASE_SIZE else REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE if desired_signal <= -BASE_SIZE else -REDUCED_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
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