#!/usr/bin/env python3
"""
Experiment #784: 4h Primary + 12h/1d HTF — Fisher Transform + ATR Volatility + HMA Trend

Hypothesis: After 500+ failed strategies, CRSI-heavy approaches are overfit. New approach:
1. Fisher Transform (Ehlers) catches reversals in bear/range markets better than RSI
2. ATR volatility expansion (ATR(7)/ATR(30) > 1.8) signals momentum ignition
3. 12h HMA(21) for smoother trend bias than EMA (less lag)
4. 1d HMA(50) for macro trend filter (only trade with macro trend)
5. Simple ADX(14) > 20 confirmation (not complex regime switching)
6. Fewer conditions = more trades (target 30-50/year on 4h)

Why this differs from failed #761-#783:
- NO Connors RSI (overused, failing recently)
- NO Choppiness Index (adds complexity, kills trades)
- NO complex regime switching (hysteresis overfitting)
- Fisher Transform is proven for crypto reversals (Ehlers literature)
- Volatility expansion filter captures breakout momentum

Strategy design:
1. 12h HMA(21) for intermediate trend bias (aligned via mtf_data)
2. 1d HMA(50) for macro trend filter (aligned via mtf_data)
3. 4h Fisher Transform (period=9) for entry timing
4. 4h ATR ratio (ATR7/ATR30) for volatility expansion
5. 4h ADX(14) > 20 for trend strength confirmation
6. 4h ATR(14) trailing stop (2.5x)
7. Discrete signals: 0.0, ±0.25, ±0.30

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive
Timeframe: 4h (target 20-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_atr_vol_hma_12h1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(series, period):
    """
    Hull Moving Average - smoother than EMA with less lag.
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    series = pd.Series(series)
    half = int(period / 2)
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = series.ewm(span=period, min_periods=period, adjust=False).mean()
    
    raw_hma = 2 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution.
    Catches reversals better than RSI in bear/range markets.
    Fisher = 0.5 * ln((1 + X) / (1 - X)) where X = 0.67 * (price - LL) / (HH - LL) - 0.33
    """
    n = len(high)
    fisher = np.full(n, np.nan)
    fisher_signal = np.full(n, np.nan)
    
    if n < period + 2:
        return fisher, fisher_signal
    
    for i in range(period, n):
        hh = np.max(high[i-period+1:i+1])
        ll = np.min(low[i-period+1:i+1])
        
        if hh - ll < 1e-10:
            fisher[i] = fisher[i-1] if i > 0 else 0.0
            fisher_signal[i] = fisher[i]
            continue
        
        price = (high[i] + low[i]) / 2.0
        x = 0.67 * ((price - ll) / (hh - ll)) - 0.33
        x = np.clip(x, -0.999, 0.999)  # prevent division by zero
        
        fisher[i] = 0.5 * np.log((1 + x) / (1 - x))
        
        if i > 0 and not np.isnan(fisher[i-1]):
            fisher_signal[i] = fisher[i-1]
        else:
            fisher_signal[i] = fisher[i]
    
    return fisher, fisher_signal

def calculate_adx(high, low, close, period=14):
    """Average Directional Index - measures trend strength."""
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2 + 1:
        return adx
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        elif minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_atr_ratio(high, low, close, short_period=7, long_period=30):
    """ATR ratio for volatility expansion detection."""
    atr_short = calculate_atr(high, low, close, short_period)
    atr_long = calculate_atr(high, low, close, long_period)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = atr_short / (atr_long + 1e-10)
    
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (4h) indicators
    fisher_4h, fisher_signal_4h = calculate_fisher_transform(high, low, period=9)
    atr_4h = calculate_atr(high, low, close, period=14)
    atr_ratio_4h = calculate_atr_ratio(high, low, close, short_period=7, long_period=30)
    adx_4h = calculate_adx(high, low, close, period=14)
    
    # Calculate and align HTF HMA for trend bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, 50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
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
    
    for i in range(400, n):
        # Skip if indicators not ready
        if np.isnan(fisher_4h[i]) or np.isnan(fisher_signal_4h[i]):
            continue
        if np.isnan(atr_4h[i]) or atr_4h[i] <= 1e-10:
            continue
        if np.isnan(atr_ratio_4h[i]):
            continue
        if np.isnan(adx_4h[i]):
            continue
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        
        # === TREND BIAS (HTF HMA) ===
        trend_12h_bullish = close[i] > hma_12h_aligned[i]
        trend_12h_bearish = close[i] < hma_12h_aligned[i]
        
        trend_1d_bullish = close[i] > hma_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # === VOLATILITY EXPANSION ===
        vol_expansion = atr_ratio_4h[i] > 1.5  # ATR(7) > 1.5x ATR(30)
        vol_normal = 1.0 <= atr_ratio_4h[i] <= 1.5
        
        # === TREND STRENGTH ===
        trend_strong = adx_4h[i] > 20
        
        # === FISHER TRANSFORM SIGNALS ===
        # Fisher crosses above -1.5 from below = long signal
        # Fisher crosses below +1.5 from above = short signal
        fisher_oversold = fisher_4h[i] < -1.5 and fisher_signal_4h[i] >= -1.5
        fisher_overbought = fisher_4h[i] > 1.5 and fisher_signal_4h[i] <= 1.5
        
        # Fisher extreme values
        fisher_extreme_low = fisher_4h[i] < -2.0
        fisher_extreme_high = fisher_4h[i] > 2.0
        
        # Fisher turning up/down
        fisher_turning_up = fisher_4h[i] > fisher_signal_4h[i] and fisher_4h[i] < 0
        fisher_turning_down = fisher_4h[i] < fisher_signal_4h[i] and fisher_4h[i] > 0
        
        desired_signal = 0.0
        
        # === LONG ENTRY CONDITIONS ===
        # Primary: 12h bullish + Fisher turning up + volatility expansion
        if trend_12h_bullish and fisher_turning_up and vol_expansion:
            if trend_strong:
                desired_signal = BASE_SIZE
            else:
                desired_signal = REDUCED_SIZE
        
        # Secondary: 1d bullish + Fisher oversold + trend alignment
        if trend_1d_bullish and fisher_oversold and trend_12h_bullish:
            desired_signal = max(desired_signal, BASE_SIZE)
        
        # Tertiary: Fisher extreme low + any bullish trend
        if fisher_extreme_low and (trend_12h_bullish or trend_1d_bullish):
            desired_signal = max(desired_signal, REDUCED_SIZE)
        
        # === SHORT ENTRY CONDITIONS ===
        # Primary: 12h bearish + Fisher turning down + volatility expansion
        if trend_12h_bearish and fisher_turning_down and vol_expansion:
            if trend_strong:
                desired_signal = -BASE_SIZE
            else:
                desired_signal = -REDUCED_SIZE
        
        # Secondary: 1d bearish + Fisher overbought + trend alignment
        if trend_1d_bearish and fisher_overbought and trend_12h_bearish:
            desired_signal = min(desired_signal, -BASE_SIZE)
        
        # Tertiary: Fisher extreme high + any bearish trend
        if fisher_extreme_high and (trend_12h_bearish or trend_1d_bearish):
            desired_signal = min(desired_signal, -REDUCED_SIZE)
        
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
        
        # === HOLD LOGIC — Maintain position if conditions intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if 12h trend intact and Fisher not overbought
                if trend_12h_bullish and fisher_4h[i] < 1.5:
                    desired_signal = BASE_SIZE if trend_strong else REDUCED_SIZE
            elif position_side < 0:
                # Hold short if 12h trend intact and Fisher not oversold
                if trend_12h_bearish and fisher_4h[i] > -1.5:
                    desired_signal = -BASE_SIZE if trend_strong else -REDUCED_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if 12h trend reverses or Fisher overbought
            if trend_12h_bearish and fisher_4h[i] > 1.0:
                desired_signal = 0.0
            # Exit if 1d trend strongly bearish
            if trend_1d_bearish and fisher_4h[i] > 0.5:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if 12h trend reverses or Fisher oversold
            if trend_12h_bullish and fisher_4h[i] < -1.0:
                desired_signal = 0.0
            # Exit if 1d trend strongly bullish
            if trend_1d_bullish and fisher_4h[i] < -0.5:
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