#!/usr/bin/env python3
"""
Experiment #003: 1d KAMA-Donchian Trend with 1w HMA Bias and Vol Filter

Hypothesis: After 2 failed regime-switching strategies (#001, #002 both negative Sharpe),
I'm trying a SIMPLER but more robust approach for daily timeframe:

1. WEEKLY HMA (1w): Stable major trend filter. Only long if price > 1w_HMA, only short if <.
   This prevents trading against the major trend (failed strategies ignored this).

2. DAILY KAMA (Kaufman Adaptive): Better than EMA for crypto's varying volatility regimes.
   KAMA adapts smoothing based on efficiency ratio - smooth in chop, responsive in trends.
   Entry when KAMA crosses price with momentum confirmation.

3. DONCHIAN BREAKOUT (20-day): Proven daily breakout signal. Entry on 20-day high/low break.

4. ATR VOLATILITY FILTER: Only trade when ATR(14)/ATR(50) is between 0.7-1.5.
   Skip trades during vol extremes (panic or complacency) - reduces whipsaw losses.

5. ASYMMETRIC SIZING: In bear regime (price < 1w_HMA), reduce short size to 0.20.
   In bull regime (price > 1w_HMA), use full long size 0.30. Protects in crashes.

6. TRAILING STOPLOSS: 2.5 * ATR(14) from highest/lowest since entry.

Why this should beat #001/#002 (both Sharpe < 0):
- Simpler logic = fewer failure points (regime switching failed twice)
- Weekly HMA = more stable than daily/4h HMA for major trend
- KAMA > EMA for crypto (adapts to vol changes)
- Vol filter = skip dangerous periods (2022 crash vol spikes)
- Asymmetric sizing = smaller losses in bear markets

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data helper (call ONCE before loop)
Target trades: 20-50/year (daily timeframe)
Position sizing: 0.20-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_donchian_1w_hma_volfilter_asym_v1"
timeframe = "1d"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts smoothing based on market efficiency (trend vs noise).
    ER = |close - close_n| / sum(|close_i - close_i-1|)
    SC = (ER * (fast_sc - slow_sc) + slow_sc)^2
    KAMA = KAMA_prev + SC * (close - KAMA_prev)
    """
    close_s = pd.Series(close)
    n = len(close_s)
    
    # Efficiency Ratio
    price_change = np.abs(close_s - close_s.shift(er_period))
    volatility = np.abs(close_s - close_s.shift(1)).rolling(window=er_period, min_periods=er_period).sum()
    er = price_change / volatility.replace(0, np.inf)
    er = er.fillna(0).clip(0, 1)
    
    # Smoothing Constant
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close_s.iloc[0]
    
    for i in range(1, n):
        kama[i] = kama[i-1] + sc.iloc[i] * (close_s.iloc[i] - kama[i-1])
    
    return kama

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max()
    lower = low_s.rolling(window=period, min_periods=period).min()
    
    return upper.values, lower.values

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period, min_periods=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=period).mean()
    rs = gain / loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_50 = calculate_atr(high, low, close, 50)
    kama_10 = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    rsi_14 = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    LONG_SIZE = 0.30
    SHORT_SIZE = 0.20  # Asymmetric: smaller shorts in bear markets
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(atr_50[i]) or atr_50[i] == 0:
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            continue
        
        if np.isnan(kama_10[i]):
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        if np.isnan(rsi_14[i]):
            continue
        
        # === WEEKLY HMA TREND BIAS ===
        bull_regime = close[i] > hma_1w_aligned[i]
        bear_regime = close[i] < hma_1w_aligned[i]
        
        # === VOLATILITY FILTER ===
        # Only trade when vol is normal (not panic or complacency)
        vol_ratio = atr_14[i] / atr_50[i]
        vol_normal = (vol_ratio > 0.7) and (vol_ratio < 1.5)
        
        # === KAMA TREND SIGNAL ===
        # Price above KAMA = bullish momentum
        # Price below KAMA = bearish momentum
        kama_bullish = close[i] > kama_10[i]
        kama_bearish = close[i] < kama_10[i]
        
        # KAMA slope (momentum confirmation)
        kama_slope_bullish = kama_10[i] > kama_10[i-5] if i > 5 else False
        kama_slope_bearish = kama_10[i] < kama_10[i-5] if i > 5 else False
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donchian_upper[i-1] if i > 0 and not np.isnan(donchian_upper[i-1]) else False
        breakout_short = close[i] < donchian_lower[i-1] if i > 0 and not np.isnan(donchian_lower[i-1]) else False
        
        # === RSI MOMENTUM FILTER ===
        # Avoid entering when RSI is at extreme (overbought long / oversold short)
        rsi_not_overbought = rsi_14[i] < 75
        rsi_not_oversold = rsi_14[i] > 25
        
        # === POSITION SIZING (Asymmetric) ===
        if bull_regime:
            current_size = LONG_SIZE
        else:
            current_size = SHORT_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # LONG ENTRY: Bull regime + Donchian breakout + KAMA bullish + vol normal + RSI ok
        if bull_regime and breakout_long and kama_bullish and kama_slope_bullish and vol_normal and rsi_not_overbought:
            new_signal = current_size
        
        # SHORT ENTRY: Bear regime + Donchian breakout + KAMA bearish + vol normal + RSI ok
        elif bear_regime and breakout_short and kama_bearish and kama_slope_bearish and vol_normal and rsi_not_oversold:
            new_signal = -current_size
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest price for long position
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest price for short position
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if weekly bias turns bearish
            if position_side > 0 and bear_regime:
                trend_reversal = True
            # Exit short if weekly bias turns bullish
            if position_side < 0 and bull_regime:
                trend_reversal = True
        
        # === RSI EXTREME EXIT ===
        rsi_exit = False
        if in_position and position_side != 0:
            # Exit long if RSI > 80 (overbought)
            if position_side > 0 and rsi_14[i] > 80:
                rsi_exit = True
            # Exit short if RSI < 20 (oversold)
            if position_side < 0 and rsi_14[i] < 20:
                rsi_exit = True
        
        # Apply stoploss or exits
        if stoploss_triggered or trend_reversal or rsi_exit:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                # New entry
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                # Exit position
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
        
        signals[i] = new_signal
    
    return signals