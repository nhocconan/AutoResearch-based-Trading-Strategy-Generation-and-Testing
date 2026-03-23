#!/usr/bin/env python3
"""
Experiment #1078: 30m Primary + 4h/1d HTF — Simplified Regime + RSI Pullback

Hypothesis: After 781 failed experiments, the key lesson for 30m timeframe is:
SIMPLER ENTRY CONDITIONS = MORE TRADES = BETTER SHARPE

Previous 30m failures (#1068, #1070, #1075) got Sharpe=0.000 because:
- Too many confluence filters (CRSI + CHOP + Session + Volume + HTF)
- Entry conditions mutually exclusive = 0 trades generated

NEW APPROACH:
1. 4h HMA21 for MACRO TREND DIRECTION (loaded ONCE via mtf_data)
2. 1d HMA50 for MAJOR BIAS (loaded ONCE via mtf_data)
3. 30m RSI(14) for ENTRY TIMING (simple: cross 35/65, not extreme 20/80)
4. ATR(14) for STOPLOSS (2.0x trailing)
5. Position size: 0.20-0.25 discrete (smaller for 30m to reduce fee drag)

Key difference from failed 30m strategies:
- RSI threshold: 35/65 (not 25/75 or 20/80) → MORE trades
- NO session filter (was killing trade count)
- NO volume filter (was killing trade count)
- HTF trend = bias only, not hard requirement

Why this should work:
- 4h trend filter prevents counter-trend trades (major edge)
- RSI 35/65 triggers on normal pullbacks (not just extremes)
- 30m timeframe = 40-80 trades/year target (optimal for this TF)
- Simpler logic = fewer conditions that can conflict

Timeframe: 30m (primary)
HTF: 4h + 1d — loaded ONCE before loop using mtf_data helper
Position Size: 0.20-0.25 discrete levels
Stoploss: 2.0x ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_hma_rsi_pullback_4h1d_atr_v1"
timeframe = "30m"
leverage = 1.0

def calculate_rsi(close, period=14):
    """
    Relative Strength Index (RSI) — momentum oscillator.
    
    Formula:
    RSI = 100 - (100 / (1 + RS))
    RS = Average Gain / Average Loss over period
    
    Signals:
    - RSI < 30 = oversold (potential long)
    - RSI > 70 = overbought (potential short)
    - For pullback entries: RSI crosses above 35 from below = long
    - RSI crosses below 65 from above = short
    """
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    # Calculate price changes
    delta = np.zeros(n)
    delta[1:] = np.diff(close)
    
    # Separate gains and losses
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    
    # Calculate average gain and loss using Wilder's smoothing
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    
    # Initial SMA for first period
    avg_gain[period] = np.mean(gains[1:period+1])
    avg_loss[period] = np.mean(losses[1:period+1])
    
    # Wilder's smoothing for rest
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i]) / period
    
    # Calculate RSI
    for i in range(period, n):
        if np.isnan(avg_gain[i]) or np.isnan(avg_loss[i]):
            continue
        if avg_loss[i] == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
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

def calculate_hma(series, period):
    """Hull Moving Average — faster and smoother than EMA."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_sma(series, period):
    """Simple Moving Average."""
    return pd.Series(series).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 4h HMA21 for intermediate trend
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA50 for major trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (30m) indicators
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    sma_50 = calculate_sma(close, 50)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    REDUCED_SIZE = 0.15
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    # Track RSI crossovers
    prev_rsi = np.full(n, 50.0)
    for i in range(1, n):
        if not np.isnan(rsi[i-1]):
            prev_rsi[i] = rsi[i-1]
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(rsi[i]) or np.isnan(atr[i]) or atr[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(sma_50[i]) or np.isnan(sma_200[i]):
            continue
        
        # === VOLATILITY REGIME (Position Sizing) ===
        atr_ratio = atr[i] / (sma_50[i] * 0.01) if sma_50[i] > 0 else 1.0
        vol_spike = atr_ratio > 2.5
        current_size = REDUCED_SIZE if vol_spike else BASE_SIZE
        
        # === MACRO TREND (4h HMA21) ===
        trend_4h_bull = close[i] > hma_4h_aligned[i]
        trend_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === MAJOR BIAS (1d HMA50) ===
        bias_1d_bull = close[i] > hma_1d_aligned[i]
        bias_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === RSI SIGNALS (Entry Timing) ===
        rsi_oversold = rsi[i] < 35.0
        rsi_overbought = rsi[i] > 65.0
        
        # RSI crossover signals (more lenient than extreme levels)
        rsi_long_cross = prev_rsi[i] < 35.0 and rsi[i] >= 35.0
        rsi_short_cross = prev_rsi[i] > 65.0 and rsi[i] <= 65.0
        
        # RSI in neutral zone (hold existing position)
        rsi_neutral_long = 35.0 <= rsi[i] <= 55.0
        rsi_neutral_short = 45.0 <= rsi[i] <= 65.0
        
        # === PRICE POSITION ===
        price_above_sma50 = close[i] > sma_50[i]
        price_below_sma50 = close[i] < sma_50[i]
        price_above_sma200 = close[i] > sma_200[i]
        price_below_sma200 = close[i] < sma_200[i]
        
        desired_signal = 0.0
        
        # === LONG ENTRY CONDITIONS ===
        # Primary: 4h trend bullish + RSI crosses above 35
        if trend_4h_bull and rsi_long_cross:
            desired_signal = current_size
        # Secondary: 4h trend bullish + RSI oversold + price above SMA200
        elif trend_4h_bull and rsi_oversold and price_above_sma200:
            desired_signal = current_size
        # Tertiary: Both HTF bullish + RSI neutral long (add to position)
        elif trend_4h_bull and bias_1d_bull and rsi_neutral_long and in_position and position_side > 0:
            desired_signal = current_size
        
        # === SHORT ENTRY CONDITIONS ===
        # Primary: 4h trend bearish + RSI crosses below 65
        if trend_4h_bear and rsi_short_cross:
            desired_signal = -current_size
        # Secondary: 4h trend bearish + RSI overbought + price below SMA200
        elif trend_4h_bear and rsi_overbought and price_below_sma200:
            desired_signal = -current_size
        # Tertiary: Both HTF bearish + RSI neutral short (add to position)
        elif trend_4h_bear and bias_1d_bear and rsi_neutral_short and in_position and position_side < 0:
            desired_signal = -current_size
        
        # === STOPLOSS CHECK (Trailing ATR 2.0x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.0 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.0 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if setup intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if 4h trend still bullish OR RSI not overbought
                if trend_4h_bull or rsi[i] < 70.0:
                    desired_signal = current_size
            elif position_side < 0:
                # Hold short if 4h trend still bearish OR RSI not oversold
                if trend_4h_bear or rsi[i] > 30.0:
                    desired_signal = -current_size
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if 4h trend reverses bearish AND RSI overbought
            if trend_4h_bear and rsi_overbought:
                desired_signal = 0.0
            # Exit long if price breaks below SMA50 strongly
            elif price_below_sma50 and rsi[i] < 45.0:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if 4h trend reverses bullish AND RSI oversold
            if trend_4h_bull and rsi_oversold:
                desired_signal = 0.0
            # Exit short if price breaks above SMA50 strongly
            elif price_above_sma50 and rsi[i] > 55.0:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            if desired_signal >= BASE_SIZE * 0.8:
                desired_signal = BASE_SIZE
            elif desired_signal >= REDUCED_SIZE * 0.8:
                desired_signal = REDUCED_SIZE
            else:
                desired_signal = REDUCED_SIZE * 0.5
        elif desired_signal < 0:
            if desired_signal <= -BASE_SIZE * 0.8:
                desired_signal = -BASE_SIZE
            elif desired_signal <= -REDUCED_SIZE * 0.8:
                desired_signal = -REDUCED_SIZE
            else:
                desired_signal = -REDUCED_SIZE * 0.5
        
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