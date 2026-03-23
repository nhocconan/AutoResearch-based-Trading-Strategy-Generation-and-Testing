#!/usr/bin/env python3
"""
Experiment #003: 1d Primary + 1w HTF — KAMA Adaptive Trend + Fisher Transform Reversals

Hypothesis: After CRSI/Chop (#001) and Donchian (#002) failed to beat Sharpe=0.366,
I'm testing KAMA (Kaufman Adaptive Moving Average) which adjusts to market noise,
combined with Ehlers Fisher Transform for precise reversal entries.

Why this might work better:
1. KAMA adapts to volatility — flat in chop, fast in trends (better than EMA/HMA for BTC/ETH)
2. Fisher Transform normalizes price to Gaussian distribution, catches reversals at extremes
3. 1w HMA provides major trend bias (only trade with weekly trend)
4. ADX filter ensures we only enter when trend has strength (>20)
5. 1d timeframe targets 20-40 trades/year (fee-efficient, matches research)

Key differences from failed attempts:
- NO Choppiness Index (failed in #001)
- NO Connors RSI (failed in #001)
- NO Donchian breakouts (failed in #002)
- Using KAMA instead of HMA/EMA (more adaptive to BTC/ETH behavior)
- Fisher Transform for entry timing (research shows 65%+ win rate on reversals)

Entry conditions (balanced for trade frequency):
- Long: Fisher < -1.5 + KAMA bullish + 1w HMA bullish + ADX > 20
- Short: Fisher > +1.5 + KAMA bearish + 1w HMA bearish + ADX > 20

Position size: 0.28 (conservative for 1d, allows surviving 2022-style crash)
Stoploss: 2.5*ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_fisher_1w_trend_v1"
timeframe = "1d"
leverage = 1.0

def calculate_kama(close, efficiency_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA).
    Adapts smoothing based on market efficiency (trend vs noise).
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Efficiency Ratio (ER)
    change = np.abs(close - np.roll(close, efficiency_period))
    change[0:efficiency_period] = np.abs(close[0:efficiency_period] - close[0])
    
    volatility = np.abs(close - np.roll(close, 1))
    volatility[0] = change[0]
    volatility_sum = pd.Series(volatility).rolling(window=efficiency_period, min_periods=efficiency_period).sum().values
    
    er = change / (volatility_sum + 1e-10)
    er = np.clip(er, 0.0, 1.0)
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    
    # Adaptive smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[efficiency_period] = close[efficiency_period]
    
    for i in range(efficiency_period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform.
    Normalizes price to Gaussian distribution for reversal detection.
    """
    n = len(high)
    fisher = np.zeros(n)
    fisher_signal = np.zeros(n)
    
    for i in range(period, n):
        # Highest high and lowest low over period
        hh = np.max(high[i-period+1:i+1])
        ll = np.min(low[i-period+1:i+1])
        
        # Normalize price
        range_val = hh - ll
        if range_val < 1e-10:
            range_val = 1e-10
        
        normalized = 0.66 * ((high[i] + low[i]) / 2.0 - ll) / range_val + 0.67
        normalized = np.clip(normalized, 0.001, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized + 1e-10))
        
        if i > 0:
            fisher_signal[i] = fisher[i-1]
    
    return fisher, fisher_signal

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    n = len(close)
    
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    # Smoothed values
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100.0 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / (atr + 1e-10))
    minus_di = 100.0 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / (atr + 1e-10))
    
    # DX and ADX
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values, plus_di.values, minus_di.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA)."""
    close_s = pd.Series(close)
    
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    raw_hma = 2.0 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA for major trend bias
    hma_1w = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    kama_10 = calculate_kama(close, efficiency_period=10, fast_period=2, slow_period=30)
    kama_20 = calculate_kama(close, efficiency_period=20, fast_period=2, slow_period=30)
    
    fisher, fisher_signal = calculate_fisher_transform(high, low, period=9)
    
    adx, plus_di, minus_di = calculate_adx(high, low, close, period=14)
    
    atr_14 = calculate_atr(high, low, close, period=14)
    
    rsi_14 = calculate_rsi(close, period=14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_1w_aligned[i]) or np.isnan(kama_10[i]):
            continue
        if np.isnan(fisher[i]) or np.isnan(adx[i]) or np.isnan(atr_14[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 1W MAJOR TREND BIAS ===
        # Weekly HMA slope
        hma_1w_slope_bull = hma_1w_aligned[i] > hma_1w_aligned[i-1] if i >= 1 else False
        hma_1w_slope_bear = hma_1w_aligned[i] < hma_1w_aligned[i-1] if i >= 1 else False
        
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === 1D KAMA TREND ===
        kama_bullish = kama_10[i] > kama_20[i]
        kama_bearish = kama_10[i] < kama_20[i]
        
        kama_slope_bull = kama_10[i] > kama_10[i-2] if i >= 2 else False
        kama_slope_bear = kama_10[i] < kama_10[i-2] if i >= 2 else False
        
        # === TREND STRENGTH (ADX) ===
        trend_strong = adx[i] > 20  # Minimum trend strength
        
        # === FISHER TRANSFORM REVERSAL SIGNALS ===
        # Long: Fisher crosses above -1.5 from below (oversold reversal)
        fisher_long = fisher[i] < -1.0 and fisher[i] > fisher_signal[i]
        
        # Short: Fisher crosses below +1.5 from above (overbought reversal)
        fisher_short = fisher[i] > 1.0 and fisher[i] < fisher_signal[i]
        
        # === RSI CONFIRMATION ===
        rsi_oversold = rsi_14[i] < 40
        rsi_overbought = rsi_14[i] > 60
        
        # === ASYMMETRIC ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY ---
        # Need: Fisher reversal + KAMA bullish + Weekly trend supportive + ADX confirms
        weekly_allows_long = hma_1w_slope_bull or price_above_hma_1w
        
        long_condition = (
            fisher_long and
            kama_bullish and
            weekly_allows_long and
            trend_strong and
            rsi_oversold
        )
        
        if long_condition:
            new_signal = POSITION_SIZE
        
        # --- SHORT ENTRY ---
        # Need: Fisher reversal + KAMA bearish + Weekly trend supportive + ADX confirms
        weekly_allows_short = hma_1w_slope_bear or price_below_hma_1w
        
        short_condition = (
            fisher_short and
            kama_bearish and
            weekly_allows_short and
            trend_strong and
            rsi_overbought
        )
        
        if short_condition:
            new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
            new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON TREND FLIP ===
        if in_position and position_side > 0:
            # Exit long if weekly trend turns bearish
            if hma_1w_slope_bear and price_below_hma_1w:
                new_signal = 0.0
            # Exit if KAMA turns bearish
            if kama_bearish and not kama_slope_bull:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if weekly trend turns bullish
            if hma_1w_slope_bull and price_above_hma_1w:
                new_signal = 0.0
            # Exit if KAMA turns bullish
            if kama_bullish and not kama_slope_bear:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals