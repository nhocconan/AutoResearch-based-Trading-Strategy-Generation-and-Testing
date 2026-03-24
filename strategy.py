#!/usr/bin/env python3
"""
Experiment #013: 1d Primary + 1w HTF — Trend-Following Mean Reversion

Hypothesis: Previous 1d strategies failed due to OVER-COMPLEXITY (too many regime filters).
This uses a SIMPLER approach proven on higher timeframes:

1. 1w HMA(21) for MAJOR trend direction (bull/bear market filter)
2. 1d RSI(14) for entry timing within trend (pullback entries)
3. 1d Donchian(20) breakout confirmation for momentum
4. ATR(14) trailing stop at 3.0x for risk management
5. Asymmetric sizing: 0.30 with 1w trend, 0.20 against

Key differences from failed attempts:
- SIMPLER logic: trend + pullback + breakout (3 filters, not 5+)
- LOOSER RSI thresholds (40/60 instead of 30/70) for more trades
- 1w HMA is very slow = fewer whipsaws in 2022 crash
- Donchian breakout ensures we catch momentum, not just mean reversion

Entry Logic:
- BULL (price > 1w HMA): RSI < 40 + price breaks Donchian(20) high → long 0.30
- BEAR (price < 1w HMA): RSI > 60 + price breaks Donchian(20) low → short 0.30
- Against trend: half size (0.20) only on extreme RSI (25/75)

Risk: 3.0x ATR trailing stop, max signal magnitude 0.35
Target: Sharpe > 0.3, trades > 30/symbol train, > 3/symbol test, DD > -40%
Trade frequency: ~40-50/year on 1d (1 per week)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_rsi_donchian_trend_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_rsi(close, period=14):
    """
    Relative Strength Index (RSI)
    RSI = 100 - 100/(1 + RS), where RS = avg_gain/avg_loss
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    # Use EMA for smoother RSI
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    for i in range(period, n):
        if avg_loss[i-1] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i-1] / avg_loss[i-1]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_hma(close, period=21):
    """
    Hull Moving Average (HMA)
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    Faster response than EMA with less lag
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, span):
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half_period)
    wma_full = wma(close, period)
    raw_hma = 2.0 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_period)
    
    return hma

def calculate_donchian(high, low, period=20):
    """
    Donchian Channel
    Upper = highest high over period
    Lower = lowest low over period
    Breakout above upper = bullish momentum
    Breakout below lower = bearish momentum
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for major trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    signals = np.zeros(n)
    
    # Position sizing
    SIZE_WITH_TREND = 0.30
    SIZE_AGAINST_TREND = 0.20
    MAX_SIZE = 0.35
    
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
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === 1W TREND BIAS ===
        # Price vs 1w HMA determines major trend
        bull_market = close[i] > hma_1w_aligned[i]
        bear_market = close[i] < hma_1w_aligned[i]
        
        # === DONCHIAN BREAKOUT DETECTION ===
        # Breakout above upper = bullish momentum
        # Breakout below lower = bearish momentum
        breakout_long = close[i] > donchian_upper[i-1] if i > 0 else False
        breakout_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # === RSI ENTRY CONDITIONS ===
        # In bull market: look for pullback longs (RSI < 40) + breakout
        # In bear market: look for bounce shorts (RSI > 60) + breakout
        rsi_oversold = rsi[i] < 40.0
        rsi_overbought = rsi[i] > 60.0
        rsi_extreme_low = rsi[i] < 25.0
        rsi_extreme_high = rsi[i] > 75.0
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        if bull_market:
            # BULL MARKET: prefer longs
            if breakout_long and rsi_oversold:
                # Strong long: with trend + pullback + breakout
                desired_signal = SIZE_WITH_TREND
            elif breakout_long and rsi[i] < 50:
                # Moderate long: with trend + breakout (RSI neutral)
                desired_signal = SIZE_WITH_TREND * 0.7
            elif rsi_extreme_low:
                # Contrarian long: extreme oversold even without breakout
                desired_signal = SIZE_AGAINST_TREND
            elif breakout_short and rsi_overbought:
                # Short against trend: only on extreme conditions
                desired_signal = -SIZE_AGAINST_TREND * 0.5
        
        elif bear_market:
            # BEAR MARKET: prefer shorts
            if breakout_short and rsi_overbought:
                # Strong short: with trend + bounce + breakout
                desired_signal = -SIZE_WITH_TREND
            elif breakout_short and rsi[i] > 50:
                # Moderate short: with trend + breakout (RSI neutral)
                desired_signal = -SIZE_WITH_TREND * 0.7
            elif rsi_extreme_high:
                # Contrarian short: extreme overbought even without breakout
                desired_signal = -SIZE_AGAINST_TREND
            elif breakout_long and rsi_oversold:
                # Long against trend: only on extreme conditions
                desired_signal = SIZE_AGAINST_TREND * 0.5
        
        # === STOPLOSS CHECK (Trailing ATR 3.0x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 3.0 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 3.0 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        # Clamp to max magnitude and discretize to reduce fee churn
        desired_signal = np.clip(desired_signal, -MAX_SIZE, MAX_SIZE)
        
        if desired_signal >= SIZE_WITH_TREND * 0.85:
            final_signal = SIZE_WITH_TREND
        elif desired_signal <= -SIZE_WITH_TREND * 0.85:
            final_signal = -SIZE_WITH_TREND
        elif desired_signal >= SIZE_AGAINST_TREND * 0.85:
            final_signal = SIZE_AGAINST_TREND
        elif desired_signal <= -SIZE_AGAINST_TREND * 0.85:
            final_signal = -SIZE_AGAINST_TREND
        elif abs(desired_signal) >= 0.10:
            final_signal = np.sign(desired_signal) * SIZE_AGAINST_TREND
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
                # Flip position
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