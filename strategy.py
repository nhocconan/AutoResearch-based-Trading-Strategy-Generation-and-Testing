#!/usr/bin/env python3
"""
Experiment #1043: 1d Primary + 1w HTF — Simplified Trend Following with Pullback Entries

Hypothesis: After 755+ failed strategies, the pattern is clear: COMPLEX regime switching
creates mutually exclusive conditions → 0 trades. The winning approach for 1d timeframe:

1. 1w HMA21 MACRO FILTER: Simple weekly trend direction. Price > HMA21 = bullish macro,
   Price < HMA21 = bearish macro. This is the ONLY HTF filter (not dual/triple HTF).

2. 1d DONCHIAN(20) BREAKOUT: Pure price action trend confirmation. Long when price breaks
   20-day high, short when breaks 20-day low. Proven across all assets.

3. 1d RSI(14) PULLBACK: Enter on pullbacks WITHIN trend, not at extremes. Long when RSI
   35-55 in uptrend, short when RSI 45-65 in downtrend. RELAXED thresholds ensure trades.

4. 1d ATR(14) TRAILING STOP: 2.5x ATR from entry high/low. Signal→0 when hit.

5. SIMPLIFIED LOGIC: Only 2-3 conditions per entry (not 5-6). This ensures 30+ trades/train.

Why 1d works:
- Lower trade frequency = less fee drag (target 20-50 trades/year)
- 1w HTF provides stable macro filter (less noise than 4h/1d HTF)
- Daily bars capture major moves without whipsaw

Timeframe: 1d (target 20-50 trades/year = 5-12 trades per symbol per year)
Position Size: 0.25-0.30 discrete levels (MAX 0.35 to survive 2022 crash)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_rsi_1w_hma_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_donchian_channels(high, low, period=20):
    """
    Donchian Channels: Highest high and lowest low over N periods
    Upper = breakout level for longs
    Lower = breakout level for shorts
    """
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_rsi(close, period=14):
    """
    Relative Strength Index
    RSI < 30 = oversold, RSI > 70 = overbought
    For pullback entries: RSI 35-55 in uptrend, 45-65 in downtrend
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
    
    gain_series = pd.Series(gain)
    loss_series = pd.Series(loss)
    
    avg_gain = gain_series.ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = loss_series.ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi[period:] = 100 - (100 / (1 + rs[period:]))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility and stoploss."""
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
    """Hull Moving Average for trend direction."""
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
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA21 for macro trend filter
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    donchian_upper, donchian_lower = calculate_donchian_channels(high, low, period=20)
    rsi_1d = calculate_rsi(close, period=14)
    atr_1d = calculate_atr(high, low, close, period=14)
    
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
        if np.isnan(rsi_1d[i]) or np.isnan(atr_1d[i]) or atr_1d[i] <= 1e-10:
            continue
        if np.isnan(hma_1w_aligned[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === MACRO TREND (1w HMA21) ===
        macro_bull = close[i] > hma_1w_aligned[i]
        macro_bear = close[i] < hma_1w_aligned[i]
        
        # === DONCHIAN TREND SIGNAL ===
        # Price above upper = strong uptrend, below lower = strong downtrend
        trend_long = close[i] >= donchian_upper[i]
        trend_short = close[i] <= donchian_lower[i]
        
        # === RSI PULLBACK ZONES (RELAXED for more trades) ===
        # In uptrend: RSI 35-60 allows entries
        # In downtrend: RSI 40-65 allows entries
        rsi_ok_long = 35 <= rsi_1d[i] <= 60
        rsi_ok_short = 40 <= rsi_1d[i] <= 65
        
        desired_signal = 0.0
        
        # === LONG ENTRIES (SIMPLIFIED - only 2 conditions) ===
        # Entry 1: Macro bull + trend long + RSI okay
        if macro_bull and trend_long and rsi_ok_long:
            desired_signal = BASE_SIZE
        # Entry 2: Macro bull + price near Donchian upper (within 2%) + RSI okay
        elif macro_bull and close[i] >= donchian_upper[i] * 0.98 and rsi_ok_long:
            desired_signal = BASE_SIZE
        # Entry 3: Macro bull + RSI oversold (reversal play)
        elif macro_bull and rsi_1d[i] < 35:
            desired_signal = REDUCED_SIZE
        
        # === SHORT ENTRIES (SIMPLIFIED - only 2 conditions) ===
        # Entry 1: Macro bear + trend short + RSI okay
        if macro_bear and trend_short and rsi_ok_short:
            desired_signal = -BASE_SIZE
        # Entry 2: Macro bear + price near Donchian lower (within 2%) + RSI okay
        elif macro_bear and close[i] <= donchian_lower[i] * 1.02 and rsi_ok_short:
            desired_signal = -BASE_SIZE
        # Entry 3: Macro bear + RSI overbought (reversal play)
        elif macro_bear and rsi_1d[i] > 65:
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
        
        # === HOLD LOGIC — Maintain position if trend intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if macro still bullish OR trend still intact
                if macro_bull or (close[i] > donchian_lower[i] and rsi_1d[i] > 30):
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if macro still bearish OR trend still intact
                if macro_bear or (close[i] < donchian_upper[i] and rsi_1d[i] < 70):
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if macro reverses bearish AND RSI overbought
            if macro_bear and rsi_1d[i] > 65:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if macro reverses bullish AND RSI oversold
            if macro_bull and rsi_1d[i] < 35:
                desired_signal = 0.0
        
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
                entry_atr = atr_1d[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Flip position
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1d[i]
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