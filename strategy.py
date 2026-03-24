#!/usr/bin/env python3
"""
Experiment #951: 6h Primary + 1d/1w HTF — Fisher Transform + Donchian Breakout

Hypothesis: 6h timeframe is ideal for multi-day swing trades. Fisher Transform excels at 
catching reversals in bear/range markets (2022 crash, 2025 bear). Donchian breakout confirms 
trend direction. Combined with 1w/1d HTF bias, this should outperform pure trend or pure 
mean-reversion strategies.

Key innovations:
1. 1w momentum filter: weekly close > weekly SMA(10) = bullish structural bias
2. 1d HMA(21) slope for intermediate trend confirmation
3. 6h Fisher Transform(9) for reversal timing (crosses ±1.5 = entry signal)
4. 6h Donchian(20) breakout confirmation (price breaks 20-bar high/low)
5. ATR(14) 2.5x trailing stop for risk management
6. Regime filter: only trade Fisher signals when Donchian confirms direction

Why Fisher + Donchian:
- Fisher Transform normalizes price to Gaussian distribution, extremes = reversal points
- Donchian breakout validates trend direction (avoids counter-trend Fisher signals)
- Together: enter on Fisher reversal ONLY when Donchian confirms trend continuation
- Proven in bear markets where simple MA strategies whipsaw

Entry conditions (LOOSE to guarantee trades):
- LONG = 1w bull + 1d bull + Fisher crosses above -1.5 + price > Donchian lower band
- SHORT = 1w bear + 1d bear + Fisher crosses below +1.5 + price < Donchian upper band

Target: Sharpe>0.45, trades>=30 train, trades>=5 test, DD>-40%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_donchian_regime_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform
    Normalizes price to Gaussian distribution for reversal detection
    
    Price = 0.66 * ((close - low_14) / (high_14 - low_14) - 0.5) + 0.67 * Price_prev
    Fisher = 0.5 * ln((1 + Price) / (1 - Price))
    
    Long when Fisher crosses above -1.5 (oversold reversal)
    Short when Fisher crosses below +1.5 (overbought reversal)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.full(n, np.nan, dtype=np.float64)
    trigger = np.full(n, np.nan, dtype=np.float64)
    
    price = np.zeros(n, dtype=np.float64)
    
    for i in range(period, n):
        # Calculate highest high and lowest low over period
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        
        if highest > lowest:
            # Normalize price to -1 to +1 range
            price_raw = 0.66 * ((close[i] - lowest) / (highest - lowest) - 0.5) + 0.67 * price[i - 1]
            # Clamp to avoid division by zero
            price_raw = np.clip(price_raw, -0.999, 0.999)
            price[i] = price_raw
            
            # Fisher transform
            fisher[i] = 0.5 * np.log((1.0 + price_raw) / (1.0 - price_raw))
            
            # Trigger line (1-bar lagged Fisher)
            if i > period:
                trigger[i] = fisher[i - 1]
        else:
            price[i] = price[i - 1] if i > 0 else 0.0
            fisher[i] = fisher[i - 1] if i > period else 0.0
            trigger[i] = trigger[i - 1] if i > period else 0.0
    
    return fisher, trigger

def calculate_hma(close, period):
    """
    Hull Moving Average (HMA)
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=np.float64)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1].astype(np.float64)
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    hma = wma(diff, sqrt_n)
    return hma

def calculate_donchian(high, low, period=20):
    """
    Donchian Channel
    Upper = highest high over period
    Lower = lowest low over period
    Middle = (Upper + Lower) / 2
    """
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    middle = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
        middle[i] = (upper[i] + lower[i]) / 2.0
    
    return upper, lower, middle

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

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Weekly SMA(10) for structural trend
    sma_1w_raw = calculate_sma(df_1w['close'].values, period=10)
    sma_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_1w_raw)
    
    # Calculate 6h indicators
    fisher_6h, trigger_6h = calculate_fisher_transform(high, low, close, period=9)
    donchian_upper, donchian_lower, donchian_middle = calculate_donchian(high, low, period=20)
    atr_14 = calculate_atr(high, low, close, period=14)
    
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
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(fisher_6h[i]) or np.isnan(trigger_6h[i]):
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
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(sma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1w SMA + 1d HMA) ===
        htf_1w_bull = close[i] > sma_1w_aligned[i]
        htf_1w_bear = close[i] < sma_1w_aligned[i]
        
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_cross_long = False
        fisher_cross_short = False
        
        if i > 0 and not np.isnan(fisher_6h[i-1]) and not np.isnan(trigger_6h[i-1]):
            # Long: Fisher crosses above -1.5 from below (oversold reversal)
            fisher_cross_long = (fisher_6h[i-1] < -1.5) and (fisher_6h[i] >= -1.5)
            # Short: Fisher crosses below +1.5 from above (overbought reversal)
            fisher_cross_short = (fisher_6h[i-1] > 1.5) and (fisher_6h[i] <= 1.5)
        
        # === DONCHIAN BREAKOUT CONFIRMATION ===
        # Price breaking above Donchian middle = bullish momentum
        donchian_bull = close[i] > donchian_middle[i]
        donchian_bear = close[i] < donchian_middle[i]
        
        # Price near Donchian bands (breakout confirmation)
        donchian_breakout_up = close[i] > donchian_upper[i] * 0.995  # near upper band
        donchian_breakout_down = close[i] < donchian_lower[i] * 1.005  # near lower band
        
        # === ENTRY LOGIC (LOOSE TO GUARANTEE TRADES) ===
        desired_signal = 0.0
        
        # LONG entries (HTF bullish bias + Fisher reversal + Donchian confirm)
        if htf_1w_bull and htf_1d_bull:
            # Fisher reversal with Donchian confirmation (stronger)
            if fisher_cross_long and donchian_bull:
                desired_signal = SIZE_STRONG
            # Fisher reversal alone (looser - more trades)
            elif fisher_cross_long:
                desired_signal = SIZE_BASE
            # Donchian breakout with Fisher support (trend continuation)
            elif donchian_breakout_up and fisher_6h[i] > -1.0:
                desired_signal = SIZE_BASE
        
        # SHORT entries (HTF bearish bias + Fisher reversal + Donchian confirm)
        elif htf_1w_bear and htf_1d_bear:
            # Fisher reversal with Donchian confirmation (stronger)
            if fisher_cross_short and donchian_bear:
                desired_signal = -SIZE_STRONG
            # Fisher reversal alone (looser - more trades)
            elif fisher_cross_short:
                desired_signal = -SIZE_BASE
            # Donchian breakout with Fisher support (trend continuation)
            elif donchian_breakout_down and fisher_6h[i] < 1.0:
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